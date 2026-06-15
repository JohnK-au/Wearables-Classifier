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

## Datasets

**PAMAP2 Physical Activity Monitoring** (UCI id 231) is used for the activities-of-daily-living analysis. It contains 100 Hz accelerometer, gyroscope, and heart rate recordings from 9 subjects performing 12 labelled activities, including everyday motion (walking, sitting, standing, cycling), household tasks (ironing, vacuuming), and exercise (running, Nordic walking, rope jumping). Activity ID 0 represents transient periods between activities and is dropped during cleaning.

> Reiss, A. & Stricker, D. (2012). PAMAP2 Physical Activity Monitoring. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/231

**RecGym** (UCI) is used for the gym exercise analysis. It contains 50 Hz IMU and capacitive sensor recordings from 10 subjects across 3 sensor positions (wrist, pocket, leg) covering 11 gym exercises plus a Null class. The wrist position is the primary focus; the leg and pocket positions are evaluated in a sensor position ablation to confirm that wrist placement is most discriminative for upper-body exercises.

> Banos, O. et al. (2014). RecGym. UCI Machine Learning Repository.

## Results

All scores use leave-one-subject-out cross-validation (LOSO). The reported figure is the mean across all 9 held-out subjects.

### PAMAP2 Activities of Daily Living (notebook 03)

| Model              | Accuracy (LOSO) | Macro F1 | Notes                                         |
|--------------------|-----------------|----------|-----------------------------------------------|
| Random Forest      | 79.5%           | 0.793    | 200 estimators, wrist acc + gyro + HR         |
| Gradient Boosting  | —               | —        | Run notebook 03 (slow: ~40 min LOSO)          |

### RecGym Gym Exercises, wrist position (notebook 04)

| Model              | Accuracy (LOSO) | Macro F1 | Notes                                         |
|--------------------|-----------------|----------|-----------------------------------------------|
| Random Forest      | 94.2%           | 0.929    | 200 estimators, wrist acc + gyro + C\_1       |
| Gaussian HMM       | —               | —        | Run notebook 04 (4 states, 500 windows/class) |

## Project layout

```
wearables-classifier/
├── data/
│   ├── raw/                  # Source files, never modified
│   ├── processed/            # Cached feature matrices (parquet)
│   └── download_data.py      # Pulls PAMAP2 into data/raw/
├── notebooks/
│   ├── 01_eda.ipynb                              # Signal exploration
│   ├── 02_feature_engineering.ipynb              # Feature construction walkthrough
│   ├── 03_activities_of_daily_living_classifier.ipynb  # PAMAP2 LOSO, ablation
│   └── 04_gym_exercise_classifier.ipynb          # RecGym LOSO, position comparison, HMM
├── src/
│   ├── config.py             # Single source of truth for all parameters
│   ├── preprocessing.py      # Load, clean, interpolate, segment_windows
│   └── features.py           # Time + frequency feature extraction
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

Run them in order: `01_eda.ipynb` to explore the signals, `02_feature_engineering.ipynb` to see how features are built, `03_activities_of_daily_living_classifier.ipynb` for the PAMAP2 LOSO analysis, and `04_gym_exercise_classifier.ipynb` for the RecGym gym exercise analysis (downloads `archive.zip` from the UCI RecGym dataset and places it in `data/raw/` before running notebook 04).

## Key findings

Random Forest with hand-crafted time and frequency features generalises well to new wearers under LOSO evaluation, demonstrating that periodic motion signatures (gait cadence, rep tempo) are consistent enough across individuals to be learned from a small cohort. Heart rate is the most discriminative channel for separating high-intensity activities such as running and rope jumping from low-intensity ones whose motion signatures alone are ambiguous. Sensor position matters substantially for gym exercise classification: wrist placement outperforms leg and pocket because the upper limb is the primary driver of most resistance exercises, and removing the capacitive channel (C\_1) from the position comparison isolates the effect cleanly.

## Notes for reviewers

- Evaluation uses leave-one-subject-out cross-validation, not a random train/test split. This is the right choice for wearables because we care whether the model generalises to a new wearer, not whether it can memorise the patterns of subjects it already saw.
- All hyperparameters live in [src/config.py](src/config.py) so experiments are easy to reproduce and easy to sweep over later.
- Notebooks are intentionally thin wrappers over the `src/` modules. The same logic that runs in tests also runs in the analysis, so nothing in the notebooks is "magic" code that disappears when you close the kernel.

## License

TBD
