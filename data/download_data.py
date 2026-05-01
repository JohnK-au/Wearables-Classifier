"""Download the PAMAP2 Physical Activity Monitoring dataset (UCI id=231).

UCI's Python API does not expose this dataset, so we download the static ZIP
directly from UCI's CDN. The archive is nested:

    pamap2+physical+activity+monitoring.zip
        |- readme.pdf
        +- PAMAP2_Dataset.zip
              |- Protocol/
              |    |- subject101.dat   (~48 MB each, space-separated text)
              |    |- subject102.dat
              |    |- ...
              |    +- subject109.dat
              +- Optional/   (extra recordings for some subjects, ignored here)

Each .dat file has 54 unnamed columns (one row per sample at 100 Hz):
    timestamp, activityID, heart_rate,
    IMU hand:   1 temperature + 3 acc16 + 3 acc6 + 3 gyro + 3 mag + 4 orient
    IMU chest:  same 17 columns
    IMU ankle:  same 17 columns

We keep only the columns that map onto our pipeline (matching the RecGym
schema we originally designed for): wrist accelerometer + gyroscope + heart
rate, plus subject_id (from the filename) and activity_id. Everything else
is dropped to keep the CSV under a few hundred megabytes.

Run from the project root:
    python data/download_data.py
"""
from __future__ import annotations

import io
import re
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd


PAMAP2_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/231/"
    "pamap2+physical+activity+monitoring.zip"
)

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "raw"
ZIP_PATH = RAW_DIR / "pamap2.zip"
OUTPUT_CSV = RAW_DIR / "pamap2.csv"

# Column names for a single .dat file, in PAMAP2's documented order.
# The 4 'orient' columns per IMU are flagged as invalid by the dataset authors,
# but we still name them so we can drop them by name later.
def _imu_columns(prefix: str) -> list[str]:
    return [
        f"{prefix}_temperature",
        f"{prefix}_acc16_x", f"{prefix}_acc16_y", f"{prefix}_acc16_z",
        f"{prefix}_acc6_x",  f"{prefix}_acc6_y",  f"{prefix}_acc6_z",
        f"{prefix}_gyro_x",  f"{prefix}_gyro_y",  f"{prefix}_gyro_z",
        f"{prefix}_mag_x",   f"{prefix}_mag_y",   f"{prefix}_mag_z",
        f"{prefix}_orient_w", f"{prefix}_orient_x", f"{prefix}_orient_y", f"{prefix}_orient_z",
    ]

PAMAP2_COLUMNS: list[str] = (
    ["timestamp", "activity_id", "heart_rate"]
    + _imu_columns("hand")
    + _imu_columns("chest")
    + _imu_columns("ankle")
)
assert len(PAMAP2_COLUMNS) == 54, f"expected 54 columns, got {len(PAMAP2_COLUMNS)}"

# Columns we keep in the final CSV. Trimmed down from 54 to a manageable set
# that matches the pipeline's CONFIG.FEATURE_COLS.
KEEP_COLUMNS: list[str] = [
    "timestamp",
    "activity_id",
    "subject_id",     # added during loading
    "heart_rate",
    "hand_acc16_x", "hand_acc16_y", "hand_acc16_z",
    "hand_gyro_x",  "hand_gyro_y",  "hand_gyro_z",
]


def _download_zip(url: str, dest: Path) -> None:
    print("Downloading PAMAP2 ZIP from UCI (~656 MB) ...")
    print(f"  URL: {url}")
    try:
        with urlopen(url) as response, open(dest, "wb") as out:
            shutil.copyfileobj(response, out)
    except URLError as exc:
        sys.exit(
            f"Download failed: {exc}. Check your internet connection or "
            "open the URL in a browser to confirm UCI is reachable."
        )
    size_mb = dest.stat().st_size / 1e6
    print(f"  Saved {size_mb:.1f} MB to {dest}")


def _open_inner_zip(outer_zip: zipfile.ZipFile) -> zipfile.ZipFile:
    """The outer archive contains a single nested ZIP plus a PDF.

    Open the inner zip in memory so we don't write 1 GB of intermediate files
    to disk just to throw them away.
    """
    inner_names = [n for n in outer_zip.namelist() if n.lower().endswith(".zip")]
    if not inner_names:
        sys.exit(f"No nested ZIP found. Outer zip contents: {outer_zip.namelist()}")
    inner_name = inner_names[0]
    print(f"  Opening nested archive: {inner_name}")
    inner_bytes = outer_zip.read(inner_name)
    return zipfile.ZipFile(io.BytesIO(inner_bytes))


def _parse_subject_dat(raw_bytes: bytes, subject_id: int) -> pd.DataFrame:
    """Parse one subject's .dat file (space-separated, no header).

    Drops rows where activity_id is 0 only at the CONFIG layer later;
    here we keep everything so EDA can see the transient label.
    Replaces 'NaN' tokens (PAMAP2 uses literal 'NaN' for missing values).
    """
    df = pd.read_csv(
        io.BytesIO(raw_bytes),
        sep=r"\s+",
        header=None,
        names=PAMAP2_COLUMNS,
        engine="python",
        na_values=["NaN"],
    )
    df["subject_id"] = subject_id
    return df[KEEP_COLUMNS]


def _extract_and_combine(zip_path: Path, output_csv: Path) -> None:
    """Extract every Protocol/subjectXXX.dat, parse, and concat into one CSV."""
    print()
    print("Extracting and combining per-subject files ...")

    frames: list[pd.DataFrame] = []
    pattern = re.compile(r".*Protocol/subject(\d+)\.dat$", re.IGNORECASE)

    with zipfile.ZipFile(zip_path) as outer:
        with _open_inner_zip(outer) as inner:
            dat_files = sorted(
                name for name in inner.namelist()
                if pattern.match(name)
            )
            if not dat_files:
                sys.exit(
                    f"No Protocol/subject*.dat files found. "
                    f"Inner zip contents: {inner.namelist()[:10]}"
                )
            print(f"  Found {len(dat_files)} subject files in Protocol/")

            for name in dat_files:
                match = pattern.match(name)
                subject_id = int(match.group(1)) if match else -1
                print(f"    parsing {name} (subject_id={subject_id}) ...")
                raw = inner.read(name)
                df = _parse_subject_dat(raw, subject_id)
                print(f"      {len(df):,} rows")
                frames.append(df)

    print("  Concatenating ...")
    combined = pd.concat(frames, ignore_index=True)
    print(f"  Total: {len(combined):,} rows, {combined.shape[1]} columns")

    print("  Writing CSV (this can take 30-60s for a few GB) ...")
    combined.to_csv(output_csv, index=False)


def download() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    _download_zip(PAMAP2_ZIP_URL, ZIP_PATH)
    _extract_and_combine(ZIP_PATH, OUTPUT_CSV)

    print(f"  Saved CSV to {OUTPUT_CSV} ({OUTPUT_CSV.stat().st_size / 1e6:.1f} MB)")

    # Schema preview so you can immediately compare against CONFIG.
    print()
    print("Schema preview (first 5 rows):")
    df = pd.read_csv(OUTPUT_CSV, nrows=5)
    print(f"  Columns ({len(df.columns)}): {list(df.columns)}")
    print("  Dtypes:")
    for col, dtype in df.dtypes.items():
        print(f"    {col}: {dtype}")
    print()
    print(df.head())

    return OUTPUT_CSV


if __name__ == "__main__":
    download()
