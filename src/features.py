"""Feature extraction from IMU + HR windows.

Each window is a 2D numpy array of shape (n_samples, n_channels). We compute
a flat dict of scalar features per window, named "<channel>_<feature>" so
the resulting feature matrix columns read like 'acc_x_mean', 'gyr_z_dom_freq'.

Two families:
    Time-domain: amplitude, variability, tempo proxies.
    Frequency-domain: rep cadence, smoothness, energy concentration.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .config import CONFIG


# ---------------------------------------------------------------------------
# Feature catalogue (used by expected_feature_count and tests)
# ---------------------------------------------------------------------------
TIME_FEATURE_NAMES: List[str] = [
    "mean", "std", "rms", "ptp", "zcr", "skew", "kurt",
]
FREQ_FEATURE_NAMES: List[str] = [
    "dom_freq", "spec_entropy", "energy_0_5hz",
]


# ---------------------------------------------------------------------------
# Helper: resolve channel names
# ---------------------------------------------------------------------------
def _resolve_axis_names(window: np.ndarray, axis_names: Optional[Sequence[str]]) -> List[str]:
    """If the caller didn't pass channel names, fall back to ch0, ch1, ..."""
    if axis_names is None:
        return [f"ch{i}" for i in range(window.shape[1])]
    elif len(axis_names) != window.shape[1]:
        raise ValueError(f"Expected {window.shape[1]} axis names, got {len(axis_names)}")
    else:
        return list(axis_names)


# ---------------------------------------------------------------------------
# Step 1: extract_time_features
# ---------------------------------------------------------------------------
def extract_time_features(
    window: np.ndarray,
    axis_names: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Per-axis time-domain features (7 per channel).

    For each channel:
        mean   resting offset (e.g. gravity on the y-axis of an accel)
        std    overall movement intensity
        rms    sqrt(mean(x^2)) -- total energy of the motion
        ptp    peak-to-peak (max - min) -- proxy for range of motion
        zcr    zero crossing rate AROUND THE MEAN (see warning)
        skew   distribution asymmetry (rep up != rep down?)
        kurt   tailedness ('spikiness' of the rep waveform)

    IMPORTANT about ZCR: accelerometer signals carry a non-zero DC offset
    from gravity. If you count crossings around 0, you might get zero
    crossings even on a wildly oscillating signal. Centre the signal on
    its mean first, then count sign changes.
    """
    axis_names = _resolve_axis_names(window, axis_names)
    feats = {}
    for i, name in enumerate(axis_names):
        sig = window[:, i].astype(float)
        feats[f"{name}_mean"] = sig.mean()
        feats[f"{name}_std"]  = sig.std()
        feats[f"{name}_rms"]  = np.sqrt(np.mean(sig ** 2))
        feats[f"{name}_ptp"]  = np.ptp(sig)

        centred = sig - sig.mean()
        signs = np.sign(centred)
        signs[signs == 0] = 1
        feats[f"{name}_zcr"] = np.sum(np.diff(signs) != 0) / max(1, len(sig) - 1)

        if sig.std() > 0:
            feats[f"{name}_skew"] = stats.skew(sig, bias=False)
            feats[f"{name}_kurt"] = stats.kurtosis(sig, bias=False)
        else:
            feats[f"{name}_skew"] = 0.0
            feats[f"{name}_kurt"] = 0.0

    return feats


# ---------------------------------------------------------------------------
# Step 2: extract_freq_features
# ---------------------------------------------------------------------------
def extract_freq_features(
    window: np.ndarray,
    sample_rate: Optional[int] = None,
    axis_names: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Per-axis frequency-domain features (3 per channel).

    For each channel:
        dom_freq        the frequency (in Hz) at which the spectrum peaks.
                        For locomotion this is the gait cadence (e.g.
                        ~1.5 Hz for walking, ~2.5 Hz for running).
        spec_entropy    Shannon entropy of the normalised power spectrum.
                        Low entropy = clean periodic motion.
                        High entropy = jittery, less regular motion.
        energy_0_5hz    sum of power in 0-5 Hz. Captures slow, deliberate
                        movement vs ballistic high-freq noise.

    Why subtract the mean before FFT: a non-zero DC offset becomes a huge
    spike at frequency 0, dwarfing everything else. 'Dominant frequency'
    becomes useless. Centring removes the spike.
    """
    sample_rate = sample_rate if sample_rate is not None else CONFIG.SAMPLE_RATE
    axis_names = _resolve_axis_names(window, axis_names)
    feats = {}

    n=window.shape[0]
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    for i, name in enumerate(axis_names):
        sig = window[:, i].astype(float)
        sig = sig - sig.mean()   # remove DC
        spectrum = np.abs(np.fft.rfft(sig))
        power = spectrum ** 2
        total_power = power.sum()

        feats[f"{name}_dom_freq"] = freqs[np.argmax(power)] if total_power > 0 else 0.0

        if total_power > 0:
            psd_norm = power / total_power
            psd_norm = psd_norm[psd_norm > 0]
            feats[f"{name}_spec_entropy"] = -np.sum(psd_norm * np.log2(psd_norm))
        else:
            feats[f"{name}_spec_entropy"] = 0.0

        feats[f"{name}_energy_0_5hz"] = power[(freqs >= 0) & (freqs <= 5)].sum()

    return feats


# ---------------------------------------------------------------------------
# Step 3: extract_features (combiner)
# ---------------------------------------------------------------------------
def extract_features(
    window: np.ndarray,
    axis_names: Optional[Sequence[str]] = None,
    sample_rate: Optional[int] = None,
) -> Dict[str, float]:
    """All time + frequency features for a single window."""
    # Just merge the dicts from extract_time_features and extract_freq_features.
    time_feats = extract_time_features(window, axis_names)
    freq_feats = extract_freq_features(window, sample_rate, axis_names)
    return {**time_feats, **freq_feats}


# ---------------------------------------------------------------------------
# Step 4: build_feature_matrix
# ---------------------------------------------------------------------------
def build_feature_matrix(
    windows: np.ndarray,
    axis_names: Optional[Sequence[str]] = None,
    sample_rate: Optional[int] = None,
) -> pd.DataFrame:
    """Build a feature DataFrame from a stack of windows.

    Input shape: (n_windows, window_size, n_channels)
    Output: DataFrame with one row per window, ~70 columns
            (7 channels x 10 features for the default config).
    """
    if windows.ndim != 3:
        raise ValueError(f"Expected windows to be 3D (n_windows, window_size, n_channels), got shape {windows.shape}")

    axis_names = list(axis_names) if axis_names is not None else list(CONFIG.FEATURE_COLS)
    
    rows = [extract_features(w, axis_names, sample_rate) for w in windows]

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 5: expected_feature_count
# ---------------------------------------------------------------------------
def expected_feature_count(n_axes: Optional[int] = None) -> int:
    """How many feature columns build_feature_matrix should produce.

    Used in tests so we don't hard-code the magic number 70 anywhere else.
    """
    n_axes = n_axes if n_axes is not None else len(CONFIG.FEATURE_COLS)
    return n_axes * (len(TIME_FEATURE_NAMES) + len(FREQ_FEATURE_NAMES))
