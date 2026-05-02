"""Sanity checks for the preprocessing and feature pipeline.

These run on synthetic data so they're fast and don't require PAMAP2 to
be downloaded. They guard the contracts the notebooks rely on:

    - segment_windows produces (n_windows, window_size, n_channels).
    - interpolate_missing actually removes NaNs.
    - build_feature_matrix produces the expected number of columns.

Mental model: tests are a CONTRACT. They lock in shapes and behaviours
your downstream code (notebooks, training script) is allowed to assume.
If you change the contract later, the tests will tell you what else
has to change too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import CONFIG
from src.features import build_feature_matrix, expected_feature_count
from src.preprocessing import clean, interpolate_missing, segment_windows


# ---------------------------------------------------------------------------
# Test fixture: synthetic data
# ---------------------------------------------------------------------------
def _make_synthetic_df(
    n_samples_per_activity: int = 400,
    n_subjects: int = 2,
    activities=(0, 1, 2),
    seed: int = 0,
) -> pd.DataFrame:
    """Build a DataFrame that LOOKS structurally like PAMAP2.

    We don't care if the values are realistic -- just that the columns
    exist, the activity ids include the transient label (0), and there
    are multiple subjects so groupby tests work.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for subject in range(n_subjects):
        for activity in activities:
            for _ in range(n_samples_per_activity):
                row = {col: float(rng.standard_normal()) for col in CONFIG.FEATURE_COLS}
                row[CONFIG.LABEL_COL] = activity
                row[CONFIG.SUBJECT_COL] = subject
                rows.append(row)
    return pd.DataFrame(rows)


def _features_implemented() -> bool:
    """Return False until src/features.py is filled in.

    Lets us skip the feature-matrix test cleanly while features.py is
    still pseudocode, instead of failing it noisily.
    """
    try:
        expected_feature_count()
        return True
    except NotImplementedError:
        return False


# ---------------------------------------------------------------------------
# Test 1: segment_windows shape contract
# ---------------------------------------------------------------------------
def test_segment_windows_output_shape():
    """segment_windows returns (n_windows, window_size, n_channels) with matching y, subjects."""
    df = _make_synthetic_df()
    df = clean(df)
    X, y, subjects = segment_windows(df)

    assert X.ndim == 3, f"Expected 3D windows, got {X.ndim}D"
    assert X.shape[1] == CONFIG.WINDOW_SIZE
    assert X.shape[2] == len(CONFIG.FEATURE_COLS)
    assert X.shape[0] > 0, "Synthetic data should produce at least one window"
    assert y.shape == (X.shape[0],)
    assert subjects.shape == (X.shape[0],)
    assert CONFIG.TRANSIENT_LABEL not in set(y.tolist()), (
        "clean() should have removed the transient label before windowing"
    )


# ---------------------------------------------------------------------------
# Test 2: interpolate_missing removes all NaNs
# ---------------------------------------------------------------------------
def test_no_nan_after_cleaning():
    """interpolate_missing leaves no NaNs in any feature column."""
    df = _make_synthetic_df()

    # Punch holes in two channels at random positions.
    rng = np.random.default_rng(42)
    cols_to_corrupt = [CONFIG.FEATURE_COLS[0], CONFIG.FEATURE_COLS[3]]
    n_holes = int(0.05 * len(df))
    for col in cols_to_corrupt:
        idx = rng.choice(df.index, size=n_holes, replace=False)
        df.loc[idx, col] = np.nan

    assert df[cols_to_corrupt].isna().any().any(), (
        "Test setup is broken: no NaNs were inserted"
    )

    cleaned = interpolate_missing(df)
    remaining = cleaned[list(CONFIG.FEATURE_COLS)].isna().sum().sum()
    assert remaining == 0, f"Expected 0 NaNs after interpolation, got {remaining}"

    # The function must not mutate the caller's DataFrame.
    assert df[cols_to_corrupt].isna().any().any(), (
        "interpolate_missing should not mutate the input DataFrame"
    )


# ---------------------------------------------------------------------------
# Test 3: feature matrix column count matches config
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not _features_implemented(),
    reason="src/features.py not yet implemented; this test will run once it is.",
)
def test_feature_matrix_column_count_matches_config():
    """build_feature_matrix produces exactly expected_feature_count() columns."""
    df = _make_synthetic_df(n_samples_per_activity=300)
    df = clean(df)
    df = interpolate_missing(df)

    X, _, _ = segment_windows(df)
    assert X.shape[0] >= 5, "Need at least 5 windows for this test to be meaningful"

    feats = build_feature_matrix(X[:5], axis_names=CONFIG.FEATURE_COLS)

    assert feats.shape == (5, expected_feature_count()), (
        f"Expected (5, {expected_feature_count()}), got {feats.shape}"
    )
    assert not feats.isna().any().any(), "Feature matrix should not contain NaNs"


# ---------------------------------------------------------------------------
# Test 4: parametrised -- overlap should change window count
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("overlap", [0.0, 0.5, 0.75])
def test_segment_windows_overlap_changes_count(overlap):
    """Higher overlap should produce more windows for the same data."""
    df = _make_synthetic_df()
    df = clean(df)

    X, _, _ = segment_windows(df, overlap=overlap)
    assert X.shape[1] == CONFIG.WINDOW_SIZE

    if overlap > 0:
        X_baseline, _, _ = segment_windows(df, overlap=0.0)
        assert X.shape[0] > X_baseline.shape[0], (
            f"overlap={overlap} should produce more windows than overlap=0.0, "
            f"got {X.shape[0]} vs {X_baseline.shape[0]}"
        )
