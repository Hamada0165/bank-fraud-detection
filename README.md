# Bank Fraud Detection

A university assignment on binary classification under extreme class
imbalance, worked on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)
Kaggle competition dataset (Vesta Corporation e-commerce transactions).

`Assignment.ipynb` covers EDA and preprocessing, a column-family analysis of
the anonymized `V`/`C`/`D`/`M`/`id` features, missingness structure, class
imbalance handling, feature engineering, and four TabNet models trained under
5-fold cross-validation. It ends with an XGBoost section and a neural-network
section that are written but were never executed — see **What did not run**
below.

## The data

- **Source**: [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection),
  a Kaggle competition. Not redistributed here — see **Getting the data**.
- **590,540** training transactions, merged from `train_transaction.csv` (394
  columns) and `train_identity.csv` (41 columns) on `TransactionID`, giving
  590,540 × 434.
- **Fraud rate: 3.4990%** (20,663 fraud / 569,877 not fraud, ~1:27.6). A
  constant "not fraud" predictor scores **96.501% accuracy** on this data —
  every accuracy number below has to be read against that.
- `TransactionDT` is seconds from an unpublished reference point; the
  community consensus (and the notebook's own `START_DATE`) is 2017-12-01.
  Under that assumption the training window is **181 days**
  (2017-12-02 → 2018-06-01), and the fraud rate is not stationary — it rises
  from 2.53% in December to 4.06% in February.
- Only **24.4%** of transactions carry an identity record, and those rows are
  much riskier (7.85% fraud vs. 2.09% without).

## Results

All four models are 5-fold `StratifiedKFold(shuffle=True, random_state=42)`
out-of-fold scores over all 590,540 training rows.

| Model | Features | ROC-AUC | PR-AUC | Log loss |
|---|---:|---:|---:|---:|
| TabNet baseline | 216 | 0.93156 | 0.61352 | — |
| **TabNet regularised** | 216 | **0.93390** | **0.65859** | 0.18455 |
| TabNet + magic features | 455 | 0.92858 | 0.62136 | 0.22432 |
| TabNet tuned (grid search, 367.7 min) | 455 | 0.93252 | 0.62939 | 0.18494 |

**The regularised model (row 2) is the best result, and it's the smaller,
untuned one.** Adding ~240 engineered features (including the "magic" UID
feature below) and running a six-hour grid search made the score *worse*, not
better — the grid search itself only spanned 0.0070 AUC across all 9
combinations tried.

**Accuracy is not a meaningful metric here and none of the numbers above use
it.** The best model's accuracy at its F1-optimal threshold is 97.42%, one
point above the 96.501% constant baseline. ROC-AUC and PR-AUC carry the
information that accuracy hides on a 3.5%-positive dataset.

### Is the cross-validation score inflated?

`TransactionDT` makes this a time-ordered dataset, but every score above comes
from a *randomly shuffled* 5-fold split, not a chronological one. Measured
during review, with a single fixed model (`HistGradientBoostingClassifier`) on
two feature sets, comparing the two validation schemes:

| Features | Scheme | ROC-AUC | PR-AUC |
|---|---|---:|---:|
| 42 raw | chronological 75/25 | 0.91059 | 0.52560 |
| 42 raw | random 5-fold OOF | 0.94674 | 0.72426 |
| 55 with UID aggregations | chronological 75/25 | 0.92939 | 0.57745 |
| 55 with UID aggregations | random 5-fold OOF | 0.96418 | 0.79664 |

The random split inflates ROC-AUC by **~0.035** regardless of whether the UID
features are included — it's the shuffle itself, not the group aggregations,
that lets the model see time-local structure it wouldn't have in production.
The **UID feature is still genuinely useful**: judged on the chronological
split, it's worth +0.0188 ROC-AUC / +0.0519 PR-AUC. It just wasn't what caused
the random-split inflation, and it didn't help the TabNet comparison above
(where it was changed at the same time as the hyperparameters). The 0.93390
headline was never re-run chronologically — TabNet training took 25–95 minutes
per fold on the hardware used — so no specific number is claimed for what it
would score under a time-aware split, only that comparable models on this data
lose about this much.

### What did not run

Cells 216–266 of the notebook — the XGBoost fit and the entire neural-network
section — have `execution_count: null` and no outputs. They were written but
never executed. **There is no XGBoost result, no neural-network result, and no
Kaggle submission** for this project; `test_preds` is accumulated across folds
but the `to_csv` calls that would write a submission file are commented out.

### Class imbalance

SMOTE was considered (three strategies are listed in the source) but the cell
that would run it is entirely commented out and `imbalanced-learn` is never
imported. Imbalance is instead handled with
`compute_class_weight('balanced')` — weights `{0: 0.5181, 1: 14.2898}` — passed
to both TabNet and the (unexecuted) Keras model.

## Attribution

The feature engineering from cell 196 onward — the `V`-column reduction, `D`
normalisation, frequency/label/group encoders, the time-consistency feature
list, and the `card1_addr1` + `(day − D1)` "UID" feature — follows public
IEEE-CIS competition kernels, transcribed with their original comments intact
while working through them:

- [`cdeotte/eda-for-columns-v-and-id`](https://www.kaggle.com/cdeotte/eda-for-columns-v-and-id)
- [`kyakovlev/ieee-fe-with-some-eda`](https://www.kaggle.com/kyakovlev/ieee-fe-with-some-eda)
- [`gemartin/load-data-reduce-memory-usage`](https://www.kaggle.com/gemartin/load-data-reduce-memory-usage)

The TabNet modelling, cross-validation, and the validation-scheme comparison
above are this project's own contribution.

## Getting the data

The competition data (~1.3GB) is not redistributed here — Kaggle's competition
rules don't permit that. To reproduce:

1. Join the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)
   competition on Kaggle and accept its rules.
2. Set up API credentials and run `datasets/download_datasets.py` — see
   [`datasets/README.md`](datasets/README.md).
3. `train_transaction.csv` and `train_identity.csv` (merged on
   `TransactionID`) are what `Assignment.ipynb` reads.

## Running the notebook

```
pip install pandas numpy scikit-learn pytorch-tabnet xgboost jupyter
jupyter notebook Assignment.ipynb
```

Cell execution counts in the committed notebook are left exactly as they were
run — including the cells that never executed — so the notebook is also a
record of what was and wasn't completed.

Two edits were made for publication, both to filesystem paths and nothing else.
The author's absolute Windows data paths in cell source were replaced with the
repo-relative `datasets/ieee dataset/`, so the notebook runs from a clone once
the data is downloaded; and a local virtualenv path that appeared in two cells'
`pip install` console output was replaced with `<venv>`. No code, no metric, no
figure, and no execution count was changed.