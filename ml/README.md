# NebulaX PS3 model baselines

This project trains one reproducible model for each Problem Statement 3 subsystem and writes the four required submission CSVs.

## Models

- **Door:** timestamp-gap segmentation and an Extra Trees classifier over electrical and motion waveform features.
- **ACV:** peer-relative car features, leave-one-case-out validation, and a supervised/unsupervised ranking blend.
- **Rail corrugation:** side-aware time/frequency features, physically valid side-swap augmentation, and a class-balanced Extra Trees model.
- **SHM:** statistical, spectral, and rainflow damage features with a log-Ridge/Extra Trees regression ensemble.

The code never modifies the supplied datasets. Feature caches live under the selected model directory.

## Setup

Python 3.10–3.13 is recommended. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Train all four models

Point `--data-root` at the organiser repository, the `PS3` directory, or `PS3/02_Datasets`:

```bash
python train.py \
  --data-root ../NebulaX-Hackathon-ProblemStatement/PS3/02_Datasets \
  --model-dir models \
  --subsystem all
```

Rail and SHM are the slow stages because they read several gigabytes and calculate signal features. Their extracted features are cached, so later runs are much faster. Use `--quick` for a smoke test with fewer trees:

```bash
python train.py --data-root ../NebulaX-Hackathon-ProblemStatement/PS3/02_Datasets --model-dir models --subsystem door,acv --quick
```

Each subsystem writes `<name>.joblib` and `<name>_metrics.json`. `training_summary.json` collects the latest validation reports. The four trained artifacts are small enough to share through Git; rebuildable feature caches under `models/cache/` remain ignored.

## Generate submission files

```bash
python predict.py \
  --input ../NebulaX-Hackathon-ProblemStatement/PS3/02_Datasets \
  --model-dir models \
  --output predictions \
  --subsystem all
```

This creates:

```text
predictions/
├── acv_predictions.csv
├── door_predictions.csv
├── rail_predictions.csv
└── shm_predictions.csv
```

Zip the CSV files at the top level of `predictions.zip`; do not put them inside an extra folder.

## Train or predict one subsystem

Use `--subsystem door`, `acv`, `rail`, or `shm`. Comma-separated values are accepted.

```bash
python train.py --data-root /path/to/02_Datasets --model-dir models --subsystem rail
python predict.py --input /path/to/02_Datasets --model-dir models --output predictions --subsystem rail
```

## Validation cautions

- Report the saved out-of-fold metrics, not training-set scores.
- Do not tune against the provided unlabeled test inputs.
- ACV has only six labelled cases, so its leave-one-case-out score has high uncertainty.
- Rail has only 14 Side I and 24 Side II training files; inspect per-class recall alongside macro F1.
- The SHM score optimizes percentage error, so all modelling happens on positive/log-scaled targets.
