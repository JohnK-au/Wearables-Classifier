# Wearables-classifier

A machine learning pipeline that classifies physical activities from wearable IMU and heart rate data.

## What this is

I built this to explore how well off-the-shelf wearable sensors can identify what someone is doing from raw motion and physiological data alone. The pipeline ingests accelerometer, gyroscope, and heart rate signals, segments them into short windows, extracts time and frequency domain features, and trains a classifier with leave-one-subject-out cross-validation so the reported scores reflect generalisation to a new wearer rather than memorisation of the training cohort.

## Why this is interesting (the biomechanics bit)

Every physical activity has a distinctive motion signature. Walking is a clean periodic gait at roughly 1.5 to 2 Hz with most energy on the vertical axis. Running pushes that cadence up to 2.5 to 3 Hz and adds a much larger peak-to-peak amplitude. Cycling shows almost no body acceleration but heavy lower-limb rotation. Sitting and standing are nearly stationary in acceleration but distinguishable by orientation. Those differences live in measurable places:

- **Time domain**: peak-to-peak amplitude, RMS, zero crossing rate (a tempo proxy), skewness and kurtosis of the signal distribution.
- **Frequency domain**: dominant frequency (gait or rep cadence), spectral entropy (smoothness vs. jitter), low-band energy concentration (slow, deliberate movement vs. ballistic).
- **Heart rate**: rises with metabolic demand, helping separate low-intensity activities (sitting, ironing) from high-intensity ones (running, rope jumping) when the motion signatures look similar.

Hand-crafting features that map onto those properties keeps the model interpretable, which matters when I want to be able to explain to a coach why a prediction was made.

## Dataset

I use the **PAMAP2 Physical Activity Monitoring** dataset from the UCI Machine Learning Repository (id 231). It contains accelerometer, gyroscope, and heart rate recordings from 9 subjects performing 12 labelled activities, including everyday motion (walking, sitting, standing, cycling), household tasks (ironing, vacuuming), and exercise (running, Nordic walking, rope jumping).

> Reiss, A. & Stricker, D. (2012). PAMAP2 Physical Activity Monitoring. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/231

Activity ID 0 represents transient periods between activities and is dropped during cleaning so the model is only ever shown labelled samples.

## Results

Placeholder. I will fill these in once I run the full pipeline end to end on the real data.

| Model              | Accuracy (LOSO) | Macro F1 | Notes                          |
|--------------------|-----------------|----------|--------------------------------|
| Random Forest      | TBD             | TBD      | Baseline, fast iteration       |
| Gradient Boosting  | TBD             | TBD      | Slower, often slightly better  |

## Project layout

```
wearables-classifier/
├── data/
│   ├── raw/                  # Source files, never modified
│   ├── processed/            # Cleaned and feature-extracted outputs
│   └── download_data.py      # Pulls PAMAP2 into data/raw/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modelling.ipynb
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   └── evaluate.py
├── models/                   # Saved .pkl artefacts (gitignored)
├── tests/
│   └── test_preprocessing.py
├── requirements.txt
└── README.md
```

## Reproduce

```bash
# 1. Clone and enter the repo
git clone <this-repo>
cd wearables-classifier

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/Scripts/activate     # Git Bash on Windows
# or:  .venv\Scripts\activate.bat (Windows cmd)
# or:  source .venv/bin/activate  (Linux / macOS)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset (~656 MB ZIP, takes a few minutes)
python data/download_data.py

# 5. Run the tests
pytest tests/

# 6. Open the notebooks
jupyter notebook
```

Run them in order: `01_eda.ipynb` to get a feel for the signals, `02_feature_engineering.ipynb` to see how features are built, then `03_modelling.ipynb` for training and evaluation.

## Notes for reviewers

- Evaluation uses leave-one-subject-out cross-validation, not a random train/test split. This is the right choice for wearables because we care whether the model generalises to a new wearer, not whether it can memorise the patterns of subjects it already saw.
- All hyperparameters live in [src/config.py](src/config.py) so experiments are easy to reproduce and easy to sweep over later.
- Notebooks are intentionally thin wrappers over the `src/` modules. The same logic that runs in tests also runs in the analysis, so nothing in the notebooks is "magic" code that disappears when you close the kernel.

## License

TBD
