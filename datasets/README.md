# Dataset Download and Preview

Scripts to download and preview two Kaggle datasets that were compared as
candidates for the assignment: **Porto Seguro's Safe Driver Prediction** and
**IEEE-CIS Fraud Detection**. Only IEEE-CIS was used for the final work — see
the [top-level README](../README.md).

## Quick Start

### Step 1: Set up Kaggle API credentials

1. Create a Kaggle account at https://www.kaggle.com/
2. Go to Account Settings -> API -> Create New Token (downloads `kaggle.json`)
3. Place `kaggle.json` at `~/.kaggle/kaggle.json` (`%USERPROFILE%\.kaggle\kaggle.json`
   on Windows)
4. Install the CLI: `pip install kaggle`

### Step 2: Download datasets

```
python datasets/download_datasets.py
```

This downloads both competitions to `datasets/porto_seguro/` and
`datasets/ieee_fraud/`.

### Step 3: Preview and compare datasets

```
python datasets/preview_datasets.py
```

Generates summary statistics and a side-by-side comparison for both datasets.

## Dataset information

### Porto Seguro's Safe Driver Prediction
- **Size**: ~595,212 training rows, ~892,816 test rows
- **Features**: 57 features (mix of binary, categorical, numerical)
- **Task**: Binary classification (predict if a driver will file an insurance claim)
- **Target**: `target` column (0 or 1)
- **Characteristics**: moderate complexity, cleaner structure, less extreme
  class imbalance

### IEEE-CIS Fraud Detection
- **Size**: 590,540 training rows
- **Features**: ~433 features after merging the transaction and identity tables
- **Task**: Binary classification (fraud detection)
- **Target**: `isFraud` column (0 or 1)
- **Characteristics**: high complexity (many features), highly imbalanced
  (3.4990% fraud), split across transaction and identity tables, many missing
  values and categorical features — this is the dataset the assignment used.

## Files in this directory

- `download_datasets.py` — downloads both competitions via the Kaggle CLI
- `preview_datasets.py` — analyzes and compares the downloaded datasets
- `README.md` — this file
