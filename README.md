# Home Credit Indonesia, Credit Default Risk (Data Scientist assessment)

Two models on the Kaggle *home-credit-default-risk* data, **LightGBM** and a **WoE Logistic Regression scorecard**.

| | OOF AUC (95% range) | Gini | KS | Fold AUC std | PSI train→test |
|---|---|---|---|---|---|
| LightGBM | 0.791 (0.788-0.793) | 0.582 | 0.441 | 0.0033 | 0.0041 |
| Scorecard (LR) | 0.757 (0.754-0.761) | 0.515 | 0.385 | 0.0034 | 0.0012 |
| LightGBM without external scores | 0.766 | 0.533 | 0.397 | 0.0025 | 0.0080 |
| Best simple rule (average of external scores) | 0.717 | | | | |

Refusing the riskiest 10% with LightGBM catches 38.0% of defaulters and lowers the approved bad rate from 8.1% to 5.6%.

`TARGET` = 1 is Kaggle's "payment difficulties" label (more than X days late on at least one of the first Y instalments; Kaggle does not publish X and Y). All figures are 5-fold out-of-fold on the 307,511 labelled clients; the Kaggle test set has no labels and is used only for drift checks. One input, credit-to-annuity (implied loan term), shifted strongly between train and test (PSI 1.63), so the models are ready for a shadow run, not for live decisions, until that shift is explained.

**Deliverables.** `notebook.ipynb` (walk-through, executed), `src/*.py` (pipeline scripts), `models/` (saved models), `HomeCredit_Assessment_Nauval.pptx` (5 slides).

## Run
```
pip install -r requirements.txt   # Python 3.12, pinned versions used for the saved outputs
export HC_RAW=/path/to/kaggle/csvs   # application_train.csv, bureau.csv, ...
bash run_all.sh
```
On Google Colab, the first code cell of `notebook.ipynb` does this for you (Kaggle download + `bash run_all.sh models`); add the Kaggle secret listed above it (a GitHub token is needed only while the repository is private).

Order is `feat_app → feat_bureau → feat_prev → feat_behav → train_lgbm → train_lgbm (HC_NO_EXT=1) → train_logreg → score → evaluate → value_monitor → model_checks → build_notebook → (execute notebook) → build_slides`.

| Script | Does |
|---|---|
| `feat_*.py` | one row per client from each table; raw CSVs are never changed |
| `train_lgbm.py`, `train_logreg.py` | 5-fold CV, metrics, saved models in `models/` |
| `score.py` | reloads the saved models and checks they reproduce the test predictions |
| `evaluate.py` | baselines, reject curve, segments, fairness, calibration, AUC bootstrap ranges |
| `value_monitor.py` | credit turned away per cut-off (no assumed margins), feature drift |
| `model_checks.py` | behaviour tests, library versions, live-test sample size |

Outputs are in `reports/` (metrics, tables, charts, Kaggle submissions) and `slides/` (charts). Seed 42 throughout.
