# imu-exercise-classifier

A machine learning pipeline that classifies weight lifting exercises from wearable IMU and heart rate data.

## What this is

I built this to explore how well off-the-shelf wearable sensors can identify what exercise someone is doing from raw motion and physiological data alone. The pipeline ingests accelerometer, gyroscope, and heart rate signals, segments them into short windows, extracts time and frequency domain features, and trains a classifier with leave-one-subject-out cross-validation so the reported scores reflect generalisation to a new wearer rather than memorisation of the training cohort.

## Why this is interesting (the biomechanics bit)

Each weight lifting movement has a distinctive motion signature. A squat is dominated by a vertical hip translation at roughly 0.5 to 1 Hz with low rotational velocity. A shoulder press shows a vertical acceleration profile concentrated in the upper limb, with a clear eccentric and concentric tempo. A bicep curl is mostly elbow flexion, so gyroscope energy spikes on a single rotational axis. Those differences live in measurable places:

- **Time domain**: peak-to-peak amplitude, RMS, zero crossing rate (a tempo proxy), skewness and kurtosis of the signal distribution.
- **Frequency domain**: dominant frequency (rep cadence), spectral entropy (smoothness vs. jitter), low-band energy concentration (slow, deliberate movement vs. ballistic).
- **Heart rate**: rises with metabolic demand, helping separate compound from isolation work.

Hand-crafting features that map onto those properties keeps the model interpretable, which matters when I want to be able to explain to a coach (or a startup) why a prediction was made.

## Dataset

I use the **RecGym** dataset from the UCI Machine Learning Repository (id 1128). It contains accelerometer, gyroscope, and heart rate recordings from subjects performing a labelled set of gym exercises.

> Koskimäki, H., Siirtola, P., & Röning, J. (2017). RecGym: Activity recognition data of gym exercises. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/1128

Activity ID 0 represents transient periods between sets and is dropped during cleaning so the model is only ever shown labelled exercise samples.

## Results

Placeholder. I will fill these in once I run the full pipeline end to end on the real data.

| Model              | Accuracy (LOSO) | Macro F1 | Notes                          |
|--------------------|-----------------|----------|--------------------------------|
| Random Forest      | TBD             | TBD      | Baseline, fast iteration       |
| Gradient Boosting  | TBD             | TBD      | Slower, often slightly better  |

## Project layout

```
imu-exercise-classifier/
├── data/
│   ├── raw/                  # Source files, never modified
│   ├── processed/            # Cleaned and feature-extracted outputs
│   └── download_data.py      # Pulls RecGym into data/raw/
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
cd imu-exercise-classifier

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/Scripts/activate     # Git Bash on Windows
# or:  .venv\Scripts\activate.bat (Windows cmd)
# or:  source .venv/bin/activate  (Linux / macOS)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset
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
