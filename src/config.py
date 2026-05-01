"""Central configuration for the imu-exercise-classifier pipeline.

Single source of truth for sampling parameters, feature columns, model
hyperparameters, and on-disk locations. Every other module reads from CONFIG
so swapping a hyperparameter only requires editing one place.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # Sampling and windowing.
    SAMPLE_RATE: int = 100         # Hz, PAMAP2 IMU sample rate.
    WINDOW_SIZE: int = 100         # Samples per window (1 second at 100 Hz).
    OVERLAP: float = 0.5            # Fraction of window shared with the next.

    # Sensor channels we extract features from. PAMAP2 wraps wrist-mounted
    # accelerometer + gyroscope under the 'hand' prefix. Heart rate is the
    # second physiological channel.
    FEATURE_COLS: List[str] = field(default_factory=lambda: [
        "hand_acc16_x", "hand_acc16_y", "hand_acc16_z",
        "hand_gyro_x",  "hand_gyro_y",  "hand_gyro_z",
        "heart_rate",
    ])

    # Column names used in the raw dataframe.
    LABEL_COL: str = "activity_id"
    SUBJECT_COL: str = "subject_id"
    TRANSIENT_LABEL: int = 0        # PAMAP2 codes 'other (transient activities)' as 0.

    # Model hyperparameters.
    MODEL_PARAMS: Dict[str, dict] = field(default_factory=lambda: {
        "random_forest": {
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_split": 2,
            "n_jobs": -1,
            "random_state": 42,
        },
        "gradient_boosting": {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 3,
            "random_state": 42,
        },
    })

    # On-disk locations.
    DATA_PATHS: Dict[str, Path] = field(default_factory=lambda: {
        "raw": PROJECT_ROOT / "data" / "raw",
        "processed": PROJECT_ROOT / "data" / "processed",
        "models": PROJECT_ROOT / "models",
    })

    @property
    def step_size(self) -> int:
        """Hop length in samples between consecutive windows."""
        return max(1, int(self.WINDOW_SIZE * (1 - self.OVERLAP)))


CONFIG = Config()