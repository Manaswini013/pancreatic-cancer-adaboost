# Pancreatic Cancer Detection Using AdaBoost on Patient Clinical Data

**A final-year AI/ML project — research / decision-support prototype only.**

> ⚠️ **This is NOT a medical device.** The model in this repo is trained on
> **synthetic** clinical data (see "About the data" below) and must never be
> used for real diagnostic or treatment decisions. Always consult a
> qualified healthcare professional for medical concerns.

---

## 1. About the data

A real, freely available dataset for this problem is the Debernardi et al.
(2020, *PLOS Medicine*) **urinary biomarker panel for pancreatic cancer**
dataset (~590 patients; biomarkers: creatinine, LYVE1, REG1B, TFF1, plus
age, sex, and plasma CA19-9). Automated access to that dataset was not
available in this environment (Kaggle requires authenticated download; the
publisher's supplementary files blocked automated fetching).

Per your instruction, this project instead uses a **synthetically generated
dataset** (`src/generate_synthetic_data.py`) that mimics the same schema,
feature ranges, class imbalance (~18% positive), and missing-value pattern
(including one biomarker with >50% missingness) as the real study, so the
full pipeline — imputation, feature selection, SMOTE, tuning, evaluation,
SHAP — can be demonstrated end-to-end with realistic numbers.

**To use this project with the real dataset instead:** download
`Debernardi et al 2020 data.csv` from Kaggle
(`johnjdavisiv/urinary-biomarkers-for-pancreatic-cancer`), place it at
`data/synthetic_pancreatic_clinical_data.csv` (or change `DATA_PATH` in
`src/train_pipeline.py`), and re-run. The pipeline auto-detects the target
column, feature types, and missingness — it does not hardcode column names,
so it should work on the real file with minimal/no changes (you would map
the 3-class `diagnosis` column to a binary PDAC-vs-not target first).

---

## 2. Folder structure

```
project/
├── README.md
├── requirements.txt
├── app.py                          # Flask prediction web app
├── templates/
│   └── index.html                 # Patient-data input form + result page
├── data/
│   └── synthetic_pancreatic_clinical_data.csv
├── src/
│   ├── generate_synthetic_data.py  # Synthetic data generator
│   ├── inspect_data.py             # Auto target/feature/missingness detector
│   └── train_pipeline.py           # Full ML pipeline (this is the core script)
├── models/
│   ├── final_adaboost_pipeline.joblib   # Saved preprocessing + trained model
│   └── model_metadata.json              # Feature schema + best params + metrics
└── outputs/
    ├── dataset_inspection_report.json
    ├── model_comparison_results.csv
    └── plots/
        ├── confusion_matrix.png
        ├── roc_curves.png
        ├── pr_curves.png
        ├── metric_comparison_bar.png
        ├── shap_feature_importance.png
        └── shap_summary_beeswarm.png
```

---

## 3. What the pipeline does (`src/train_pipeline.py`)

1. **Load + auto-inspect** (`inspect_data.py`) — detects the target column,
   numeric vs categorical features, ID-like columns to drop, missingness,
   and class balance, without hardcoding column names.
2. **Train/test split** — stratified 80/20 split, held out **before** any
   preprocessing is fit.
3. **Preprocessing** — median imputation + standard scaling for numeric
   features; most-frequent imputation + one-hot encoding for categoricals,
   via `ColumnTransformer`.
4. **Feature selection** — `SelectKBest` (ANOVA F-test), with `k` tuned by
   grid search.
5. **Class-imbalance handling** — `SMOTE` oversampling.
6. **No data leakage** — preprocessing, feature selection, and SMOTE are all
   placed *inside* a single `imblearn.pipeline.Pipeline`, which is what is
   passed to `GridSearchCV`/cross-validation. This guarantees every
   imputation statistic, scaling statistic, feature-selection score, and
   synthetic SMOTE sample is (re)computed from the **training fold only** on
   every split — never from a validation fold or the final test set. At
   inference time, `imblearn` pipelines automatically skip the SMOTE step
   (it is a `fit_resample`-only, training-time step), so predictions on new
   patients are unaffected by it.
7. **Stratified 5-fold CV + `GridSearchCV`** over AdaBoost hyperparameters
   (`n_estimators`, `learning_rate`, base-estimator depth, `k` features).
8. **Comparison models** — Logistic Regression, Decision Tree, Random
   Forest, each wrapped in the same leakage-safe pipeline structure.
9. **Metrics** — Accuracy, Precision, Recall/Sensitivity, F1, Specificity,
   ROC-AUC, all computed on the held-out test set only.
10. **Plots** — confusion matrix, ROC curves, precision-recall curves, and a
    metric-comparison bar chart across all 4 models.
11. **SHAP explainability** — feature-importance bar chart and beeswarm
    summary plot for the tuned AdaBoost model.
12. **Persistence** — the full fitted pipeline (preprocessing + feature
    selection + AdaBoost) is saved with `joblib`, plus a metadata JSON
    (feature schema, best hyperparameters, test metrics) used by the Flask
    app.

---

## 4. Actual results from this run (synthetic data)

Dataset: 1,200 synthetic patients, 18.0% cancer-positive (realistic
imbalance), 7 numeric + 3 categorical features, missingness from ~1.5% up
to ~50.6% (an optional biomarker), auto-detected — see
`outputs/dataset_inspection_report.json`.

**Best AdaBoost hyperparameters** (via `GridSearchCV`, scoring = ROC-AUC):
```
classifier__estimator     = DecisionTreeClassifier(max_depth=1)
classifier__learning_rate = 0.5
classifier__n_estimators  = 200
feature_select__k         = 8
```

**Stratified 5-fold CV ROC-AUC (tuned AdaBoost):** 0.9702 ± 0.0096

**Held-out test-set comparison (240 patients, never seen during tuning):**

| Model | Accuracy | Precision | Recall (Sensitivity) | F1 | Specificity | ROC-AUC |
|---|---|---|---|---|---|---|
| **AdaBoost (tuned)** | 0.9125 | 0.720 | 0.837 | 0.774 | 0.929 | **0.967** |
| Random Forest | 0.9167 | 0.780 | 0.744 | 0.762 | 0.954 | 0.953 |
| Logistic Regression | 0.9000 | 0.694 | 0.791 | 0.739 | 0.924 | 0.949 |
| Decision Tree | 0.8708 | 0.600 | 0.837 | 0.699 | 0.878 | 0.891 |

AdaBoost achieved the best ROC-AUC and the best sensitivity (recall) tied
with Decision Tree, which matters clinically since missing an actual cancer
case (false negative) is usually costlier than a false alarm. Random Forest
had marginally higher raw accuracy/specificity but lower sensitivity.

SHAP confirmed that the four urinary biomarkers used in the real published
study — **REG1B, TFF1, LYVE1, CA19-9** — are also the top predictors in this
synthetic-data model, in that order, followed by age. This is expected
because the synthetic generator was built to encode exactly this kind of
signal.

All numbers above are reproducible by re-running `src/train_pipeline.py`.

---

## 5. How to run locally

```bash
# 1. Clone/copy this project folder, then create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) regenerate the synthetic dataset
python3 src/generate_synthetic_data.py

# 4. Run the full training pipeline (prints inspection report + metrics,
#    writes plots to outputs/plots/, saves the model to models/)
python3 src/train_pipeline.py

# 5. Launch the Flask prediction app
python3 app.py
# then open http://127.0.0.1:5000 in your browser
```

---

## 6. Flask interface

`app.py` + `templates/index.html` provide a simple web form:

- Numeric fields for each biomarker/age (blank = "not measured", handled by
  the same imputer used during training — no crashes on missing input).
- Dropdowns for categorical fields (sex, sample origin, cohort).
- On submit: shows predicted class, probability of cancer, and the top
  SHAP-based feature contributions for that specific patient.
- A visible disclaimer banner reminds users this is a research prototype,
  not a diagnostic tool.

Tested locally (via `curl`) with a high-biomarker synthetic patient
(→ 65.24% predicted probability, positive) and a low-biomarker patient
(→ 18.61% predicted probability, negative), plus a request with all fields
blank (→ handled gracefully via imputation, no server error).

---

## 7. Limitations & honesty notes

- **Synthetic data**: absolute performance numbers (ROC-AUC ~0.97) reflect
  patterns intentionally built into the generator, not a real clinical
  validation. Do not quote these numbers as evidence of real-world
  diagnostic accuracy.
- **Small "optional biomarker" missingness pattern** (REG1A, CA19-9) was
  simulated as reasonably-random for demonstration; real-world missingness
  mechanisms can be more complex (e.g., missing-not-at-random tied to
  clinical site).
- Class imbalance handling (SMOTE) inside CV means each fold's synthetic
  minority samples differ; results are stable (CV std ≈ 0.01) but should
  always be reported with variance, not a single number.
