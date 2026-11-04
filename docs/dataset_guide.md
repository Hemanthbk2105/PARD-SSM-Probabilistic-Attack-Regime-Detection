# Dataset Download Guide

This guide explains how to download and prepare the datasets used in PARD-SSM.

---

## CICIDS2017

**Source:** Canadian Institute for Cybersecurity
**URL:** https://www.unb.ca/cic/datasets/ids-2017.html
**Size:** ~1.2 GB (CSV files)

### Steps

1. Visit the URL above and request download access (free registration)
2. Download the "MachineLearningCSV.zip" file
3. Unzip into `data/raw/cicids2017/`

Your structure should look like:
```
data/raw/cicids2017/
    Monday-WorkingHours.pcap_ISCX.csv
    Tuesday-WorkingHours.pcap_ISCX.csv
    Wednesday-workingHours.pcap_ISCX.csv
    Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
    Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
    Friday-WorkingHours-Morning.pcap_ISCX.csv
    Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
    Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
```

4. Preprocess:
```bash
python src/data_processing/dataset_loader.py --dataset cicids2017 \
    --input data/raw/cicids2017/ \
    --output data/processed/ --sample 0
```

---

## UNSW-NB15

**Source:** UNSW Canberra Cyber
**URL:** https://research.unsw.edu.au/projects/unsw-nb15-dataset
**Size:** ~100 MB

### Steps

1. Download the CSV files from the URL above
2. Place in `data/raw/unswnb15/`

```
data/raw/unswnb15/
    UNSW-NB15_1.csv
    UNSW-NB15_2.csv
    UNSW-NB15_3.csv
    UNSW-NB15_4.csv
```

3. Preprocess:
```bash
python src/data_processing/dataset_loader.py --dataset unswnb15 \
    --input data/raw/unswnb15/ \
    --output data/processed/ --sample 0
```

---

## Using Synthetic Data (No Download Required)

For quick testing without downloading datasets, you can generate synthetic data:

```python
import numpy as np

# Simulate 3 regimes × 1000 time steps × 10 features
np.random.seed(42)
X_normal = np.random.randn(1000, 10) * 0.5
X_attack = np.random.randn(500, 10) * 2.0 + 3.0
X_exfil  = np.random.randn(500, 10) * 1.0 - 2.0

X = np.vstack([X_normal, X_attack, X_exfil])
y = np.array([0]*1000 + [1]*500 + [2]*500)

np.save("data/processed/synthetic_features.npy", X)
np.save("data/processed/synthetic_labels.npy", y)
```

Then run the experiment:
```bash
python experiments/run_switching.py \
    --data data/processed/synthetic_features.npy \
    --labels data/processed/synthetic_labels.npy \
    --regimes 3
```


## Preserving rows for temporal evaluation

The loaders concatenate CSV files in sorted filename order. Set `--sample 0` to disable random sampling and retain all rows in that deterministic order. This does not by itself prove that the result is chronological: check the dataset release's capture schedule and timestamp ordering before using a temporal train/test split.


## Leakage-safe chronological benchmark

After creating feature and label arrays with `--sample 0`, verify that the rows are ordered by the capture timestamps and that the same row filtering was applied to both arrays. Then run:

```bash
python experiments/chronological_network_benchmark.py \
  --data data/processed/cicids2017_features.npy \\
  --labels data/processed/cicids2017_labels.npy \\
  --test-ratio 0.2 \\
  --output cicids2017_chronological_report.json
```

The benchmark fits RobustScaler and PCA on the training segment only, then reuses them for the test segment. It compares PARD-SSM with a Gaussian HMM. It requires local dataset arrays and does not run on the synthetic CI data. Review the reported train/test regime lists: if a regime appears only in the test segment, this should be reported as an unseen-regime limitation rather than hidden.
