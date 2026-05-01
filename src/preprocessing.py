"""Loading, cleaning, and windowing of raw IMU + HR data.

This module turns a streaming sensor recording (a long, possibly noisy,
variable-length DataFrame) into a clean stack of fixed-size windows that
ML models can train on.

Four steps, in order:
    1. load_data           -> get raw data off disk
    2. clean               -> drop transient rows between labelled activities
    3. interpolate_missing -> patch over sensor dropouts
    4. segment_windows     -> chop into fixed-size windows for the model
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from .config import CONFIG


# ---------------------------------------------------------------------------
# Step 1: load_data
# ---------------------------------------------------------------------------
def load_data(path: str | Path) -> pd.DataFrame:
    """Load the raw dataset from CSV or Parquet."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file extension: {path.suffix}")



# ---------------------------------------------------------------------------
# Step 2: clean
# ---------------------------------------------------------------------------
def clean(
    df: pd.DataFrame,
    label_col: str = CONFIG.LABEL_COL,
    transient_label: int = CONFIG.TRANSIENT_LABEL,
) -> pd.DataFrame:
    """Drop transient periods (between activities) and reset the index.

    Why this matters: PAMAP2 labels transient periods between activities as
    activity 0. If we left them in, the model would learn 'low or transitional
    movement = transient', and 'transient' isn't an activity we care about
    classifying. We want the classifier to only ever see real, labelled
    activity samples.

    Analogy: editing out the silences between songs before you count beats
    per song. The silences are real, but they're not what you're measuring.

    Why reset the index: after filtering, pandas keeps the original row
    numbers (0, 1, 4, 7, 8, ...). That's a footgun for later groupby and
    iloc operations downstream. Resetting to (0, 1, 2, 3, ...) avoids it.
    """
    # Check that the label column exists in the DataFrame. If not, raise a KeyError
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found in DataFrame.")

    # Filter out rows where the label column equals the transient label
    mask = df[label_col] != transient_label
    filtered_df = df[mask]

    # Reset the index of the filtered DataFrame, dropping the old index
    return filtered_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 3: interpolate_missing
# ---------------------------------------------------------------------------
def interpolate_missing(
    df: pd.DataFrame,
    cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Forward fill, then linear interpolation, then back fill leading NaNs.

    Why three layers? Each handles a different gap pattern:

        ffill         tiny dropouts (a few samples). Carries the last
                      known value forward. Fast, doesn't smear noise.

        interpolate   longer gaps. Draws a straight line between the
                      last good and next good sample so we don't get a
                      flat-line artefact in the FFT later.

        bfill         NaNs at the very start of the recording, where
                      ffill has nothing to propagate from.

    Order matters: ffill first (cheap, handles the common case), then
    interpolate (handles the leftovers), then bfill (mops up the edges).
    """

    # Default `cols` to CONFIG.FEATURE_COLS if the caller didn't specify.
    cols = list(cols) if cols is not None else list(CONFIG.FEATURE_COLS)

    # Make a copy of the DataFrame so we don't mutate the caller's data.
    out = df.copy()

    # Interpolate data via forward fill, then linear interpolation, then back fill leading NaNs.
    out[cols] = (out[cols]
        .ffill()
        .interpolate(method="linear", limit_direction="both")
        .bfill()
    )
    return out

# ---------------------------------------------------------------------------
# Step 4: segment_windows
# ---------------------------------------------------------------------------
def segment_windows(
    df: pd.DataFrame,
    label_col: str = CONFIG.LABEL_COL,
    subject_col: str = CONFIG.SUBJECT_COL,
    feature_cols: Optional[Iterable[str]] = None,
    window_size: Optional[int] = None,
    overlap: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sliding-window segmentation grouped by (subject, activity).

    Why windows at all: the model needs fixed-shape inputs. A 1-second
    window at 100 Hz is always 100 samples per channel, regardless of
    whether the underlying recording was 30 seconds or 3 minutes long.

    Why group by (subject, activity) before sliding: if we slid windows
    over the whole DataFrame, a single window could straddle two
    different activities (e.g. walking -> standing -> sitting). That
    would create a label-noisy training sample where the window is
    half-walking, half-standing but tagged with a single label. Grouping
    first guarantees every window belongs to exactly one (subject,
    activity) pair.

    Why overlap (typically 50%): each new window starts halfway through the
    previous one, so we get more training samples without much information
    leakage. With overlap=0, you might only get 30 windows from a 30s
    recording at 1s/window. With overlap=0.5, you get ~59.

    Returns
    -------
    X : ndarray, shape (n_windows, window_size, n_channels)
    y : ndarray, shape (n_windows,)            label per window
    subjects : ndarray, shape (n_windows,)     subject id per window
    """

    # Resolve defaults from CONFIG for anything the caller left as None (feature_cols, window_size, overlap).
    feature_cols = list(feature_cols) if feature_cols is not None else list(CONFIG.FEATURE_COLS)
    window_size = window_size or CONFIG.WINDOW_SIZE
    overlap = overlap if overlap is not None else CONFIG.OVERLAP

    # Compute the step size (hop length) in samples from the overlap:
    #     step = window_size * (1 - overlap), but at least 1.
    #       With overlap=0.0 -> step = window_size       (no overlap)
    #       With overlap=0.5 -> step = window_size / 2   (50% overlap)
    #       With overlap=0.75 -> step = window_size / 4  (75% overlap)
    step = max(1, int(window_size * (1 - overlap)))
    
    # Validate that all required columns (feature_cols + label_col + subject_col) are present in df. 
    # Raise a KeyError listing the missing ones.
    missing = [c for c in feature_cols + [label_col, subject_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    
    # Initialise three empty Python lists: X_list, y_list, subj_list.
    X_list, y_list, subj_list = [], [], []

    # Iterate over groups using: df.groupby([subject_col, label_col], sort=False)
    for (subj, act), group in df.groupby([subject_col, label_col], sort=False):
        # Pull the feature columns out as a 2D numpy array (group[feature_cols].to_numpy(dtype=float)).
        arr = group[feature_cols].to_numpy(dtype=float)

        # If the group is shorter than window_size, skip it — there isn't enough data for even one window.
        if len(arr) < window_size:
            continue

        # Slide a window from start=0 to len(arr) - window_size stepping by `step`. For each window:
        for start in range(0, len(arr) - window_size + 1, step):
            # - slice arr[start : start + window_size]
            X_list.append(arr[start : start + window_size])
            # - append `activity` to y_list
            y_list.append(act)
            # - append `subject` to subj_list
            subj_list.append(subj)

    # If X_list is empty (all groups too short), return three empty arrays with the right shapes so 
    # downstream callers don't have to special-case "no data":
    #     (np.empty((0, window_size, len(feature_cols))),
    #      np.empty((0,)),
    #      np.empty((0,)))
    if not X_list:
        return (
            np.empty((0, window_size, len(feature_cols))),
            np.empty((0,)),
            np.empty((0,)),
        )
    
    # Otherwise: np.stack X_list into a 3D array, convert y_list and subj_list to numpy arrays with np.asarray, and return the tuple.
    return np.stack(X_list), np.asarray(y_list), np.asarray(subj_list)