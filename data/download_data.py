"""Download the RecGym dataset from the UCI ML Repository (id=1128).

Saves the full table (features + targets) to data/raw/recgym.csv and dumps
the metadata to data/raw/recgym_metadata.txt for reference.

Run from the project root:
    python data/download_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


RECGYM_ID = 1128
HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "raw"


def download() -> Path:
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        sys.exit(
            "Missing dependency: ucimlrepo. Install it with `pip install ucimlrepo`."
        )

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching RecGym (UCI id={RECGYM_ID}) ...")
    repo = fetch_ucirepo(id=RECGYM_ID)

    features = repo.data.features
    targets = repo.data.targets

    df = features.copy()
    if targets is not None:
        for col in targets.columns:
            df[col] = targets[col].values

    out_path = RAW_DIR / "recgym.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} rows and {df.shape[1]} columns to {out_path}")

    meta_path = RAW_DIR / "recgym_metadata.txt"
    meta_path.write_text(
        "RecGym (UCI id=1128) metadata\n"
        "=================================\n\n"
        f"{repo.metadata}\n\n"
        "Variables\n---------\n"
        f"{repo.variables}\n",
        encoding="utf-8",
    )
    print(f"Saved metadata to {meta_path}")
    print("\nColumn preview:")
    print(df.head())
    return out_path


if __name__ == "__main__":
    download()
