"""
train_pipeline.py
---------------------------------------------------------------
Pancreatic Cancer Detection Using AdaBoost on Patient Clinical Data.

End-to-end pipeline:
  1. Load + auto-inspect data (target, feature types, missingness, imbalance)
  2. Train/test split (stratified, held out BEFORE any fitting)
  3. Preprocessing (imputation + scaling + one-hot encoding) inside a
     scikit-learn Pipeline / ColumnTransformer, fit only on training folds
  4. Feature selection (SelectKBest) fit only on training folds
  5. Class-imbalance handling (SMOTE) applied only to training folds,
     inside an imblearn Pipeline, so the test set / CV validation folds
     never see synthetic oversampled rows -> NO DATA LEAKAGE
  6. Stratified 5-fold CV + GridSearchCV hyperparameter tuning for AdaBoost
  7. Comparison against Logistic Regression, Decision Tree, Random Forest
  8. Full metrics: Accuracy, Precision, Recall, F1, Specificity, ROC-AUC
  9. Plots: confusion matrix, ROC curve, PR curve (best model)
 10. SHAP explainability for the final AdaBoost model
 11. Persist the final trained pipeline (preprocessing + model) to disk

NOTE ON DATA LEAKAGE
Feature selection and SMOTE are placed INSIDE the imblearn Pipeline object
that is passed to GridSearchCV. This guarantees that, for every CV split,
imputation statistics, scaling statistics, SelectKBest scores, and SMOTE
synthetic samples are all computed from the training fold ONLY, and never
from the validation fold or the final held-out test set.
---------------------------------------------------------------
"""

import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from inspect_data import classify_columns, guess_target_column, inspect, print_report

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DATA_PATH = "/home/claude/project/data/synthetic_pancreatic_clinical_data.csv"
OUTPUT_DIR = Path("/home/claude/project/outputs")
PLOTS_DIR = OUTPUT_DIR / "plots"
MODELS_DIR = Path("/home/claude/project/models")
for d in (OUTPUT_DIR, PLOTS_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def build_preprocessor(numeric_cols, categorical_cols):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])
    return preprocessor


def evaluate_model(name, model, X_test, y_test, results_list):
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "specificity": specificity_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    results_list.append(metrics)
    return y_pred, y_proba, metrics


def main():
    # ---------------------------------------------------------------
    # 1. Load + inspect
    # ---------------------------------------------------------------
    report, df = inspect(DATA_PATH)
    print_report(report)

    target_col = report["target_col"]
    id_cols = report["id_cols"]
    numeric_cols = report["numeric_cols"]
    categorical_cols = report["categorical_cols"]

    with open(OUTPUT_DIR / "dataset_inspection_report.json", "w") as f:
        json.dump({k: v for k, v in report.items() if k != "dtypes"}, f, indent=2, default=str)

    X = df.drop(columns=[target_col] + id_cols)
    y = df[target_col].astype(int)

    # ---------------------------------------------------------------
    # 2. Train/test split -- held out BEFORE any preprocessing is fit
    # ---------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train class balance:\n{y_train.value_counts(normalize=True)}")
    print(f"Test class balance:\n{y_test.value_counts(normalize=True)}")

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    # Number of features to keep after preprocessing (one-hot expands categoricals)
    n_features_before_selection = None  # determined dynamically inside pipeline via 'all'

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # ---------------------------------------------------------------
    # 3-6. AdaBoost pipeline: preprocess -> feature select -> SMOTE -> AdaBoost
    #      All steps are inside ONE imblearn Pipeline so GridSearchCV / CV
    #      refits preprocessing, feature selection AND SMOTE fresh on every
    #      training fold. No leakage into validation folds or test set.
    # ---------------------------------------------------------------
    ada_pipeline = ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("feature_select", SelectKBest(score_func=f_classif, k="all")),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("classifier", AdaBoostClassifier(random_state=RANDOM_STATE)),
    ])

    param_grid = {
        "feature_select__k": [5, 8, "all"],
        "classifier__n_estimators": [50, 100, 200],
        "classifier__learning_rate": [0.01, 0.1, 0.5, 1.0],
        "classifier__estimator": [DecisionTreeClassifier(max_depth=1),
                                   DecisionTreeClassifier(max_depth=2)],
    }

    grid_search = GridSearchCV(
        ada_pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=0,
    )

    print("\nRunning GridSearchCV for AdaBoost (Stratified 5-Fold CV)...")
    grid_search.fit(X_train, y_train)

    print(f"\nBest CV ROC-AUC: {grid_search.best_score_:.4f}")
    print(f"Best params: {grid_search.best_params_}")

    best_ada_pipeline = grid_search.best_estimator_

    # Cross-validated scores of the tuned model (for reporting stability)
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(best_ada_pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"5-Fold CV ROC-AUC scores (best AdaBoost config): {cv_scores}")
    print(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # ---------------------------------------------------------------
    # 7. Comparison models -- same leakage-safe structure
    #    (preprocess -> feature_select -> SMOTE -> classifier)
    # ---------------------------------------------------------------
    comparison_models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=5),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=200),
    }

    fitted_pipelines = {"AdaBoost (tuned)": best_ada_pipeline}
    results_list = []

    # Evaluate tuned AdaBoost on test set
    evaluate_model("AdaBoost (tuned)", best_ada_pipeline, X_test, y_test, results_list)

    for name, clf in comparison_models.items():
        pipe = ImbPipeline(steps=[
            ("preprocessor", build_preprocessor(numeric_cols, categorical_cols)),
            ("feature_select", SelectKBest(score_func=f_classif,
                                            k=grid_search.best_params_["feature_select__k"])),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("classifier", clf),
        ])
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe
        evaluate_model(name, pipe, X_test, y_test, results_list)

    results_df = pd.DataFrame(results_list).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    print("\n" + "=" * 70)
    print("MODEL COMPARISON ON HELD-OUT TEST SET")
    print("=" * 70)
    print(results_df.to_string(index=False))
    results_df.to_csv(OUTPUT_DIR / "model_comparison_results.csv", index=False)

    # ---------------------------------------------------------------
    # 8-9. Plots for the best model overall (by test ROC-AUC)
    # ---------------------------------------------------------------
    best_model_name = results_df.iloc[0]["model"]
    best_pipeline = fitted_pipelines[best_model_name]
    print(f"\nBest model on held-out test set: {best_model_name}")

    y_pred_best = best_pipeline.predict(X_test)
    y_proba_best = best_pipeline.predict_proba(X_test)[:, 1]

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 5))
    cm = confusion_matrix(y_test, y_pred_best)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Cancer", "Cancer"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix - {best_model_name}")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # ROC curve - all models overlaid
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, pipe in fitted_pipelines.items():
        proba = pipe.predict_proba(X_test)[:, 1]
        RocCurveDisplay.from_predictions(y_test, proba, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_title("ROC Curves - Model Comparison")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)

    # Precision-Recall curve - all models overlaid
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, pipe in fitted_pipelines.items():
        proba = pipe.predict_proba(X_test)[:, 1]
        PrecisionRecallDisplay.from_predictions(y_test, proba, name=name, ax=ax)
    ax.set_title("Precision-Recall Curves - Model Comparison")
    ax.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pr_curves.png", dpi=150)
    plt.close(fig)

    # Bar chart of metrics across models
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_df = results_df.set_index("model")[
        ["accuracy", "precision", "recall_sensitivity", "f1", "specificity", "roc_auc"]
    ]
    plot_df.plot(kind="bar", ax=ax)
    ax.set_title("Metric Comparison Across Models")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "metric_comparison_bar.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------
    # 10. SHAP explainability for the tuned AdaBoost model
    # ---------------------------------------------------------------
    ada_only = fitted_pipelines["AdaBoost (tuned)"]
    pre = ada_only.named_steps["preprocessor"]
    fsel = ada_only.named_steps["feature_select"]
    clf = ada_only.named_steps["classifier"]

    # Build transformed feature names
    num_feature_names = numeric_cols
    ohe = pre.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = list(ohe.get_feature_names_out(categorical_cols)) if categorical_cols else []
    all_feature_names = np.array(num_feature_names + cat_feature_names)

    selected_mask = fsel.get_support()
    selected_feature_names = all_feature_names[selected_mask]

    X_test_transformed = pre.transform(X_test)
    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()
    X_test_selected = fsel.transform(X_test_transformed)
    X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_feature_names)

    print("\nComputing SHAP values for AdaBoost (tuned) model...")
    explainer = shap.TreeExplainer(clf) if False else shap.Explainer(clf.predict_proba, X_test_selected_df.sample(
        min(100, len(X_test_selected_df)), random_state=RANDOM_STATE))
    shap_values = explainer(X_test_selected_df.sample(min(200, len(X_test_selected_df)), random_state=RANDOM_STATE))

    # SHAP summary (bar) plot - class 1 (cancer)
    fig = plt.figure(figsize=(8, 6))
    try:
        sv_class1 = shap_values[..., 1] if shap_values.values.ndim == 3 else shap_values
        shap.plots.bar(sv_class1, show=False, max_display=10)
    except Exception as e:
        print(f"SHAP bar plot fallback due to: {e}")
        shap.plots.bar(shap_values, show=False, max_display=10)
    plt.title("SHAP Feature Importance - AdaBoost (Cancer class)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_feature_importance.png", dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6))
    try:
        shap.plots.beeswarm(sv_class1, show=False, max_display=10)
    except Exception as e:
        print(f"SHAP beeswarm fallback due to: {e}")
        shap.plots.beeswarm(shap_values, show=False, max_display=10)
    plt.title("SHAP Summary (Beeswarm) - AdaBoost (Cancer class)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_summary_beeswarm.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------
    # 11. Persist final trained pipeline (preprocessing + feature selection +
    #     classifier). NOTE: SMOTE is a training-time-only step; imblearn
    #     Pipelines automatically skip resampling steps at inference/.predict
    #     time, so the saved pipeline is safe to use directly on new patients.
    # ---------------------------------------------------------------
    final_model_path = MODELS_DIR / "final_adaboost_pipeline.joblib"
    joblib.dump(best_ada_pipeline, final_model_path)
    print(f"\nSaved final trained pipeline to: {final_model_path}")

    # Save metadata needed by the Flask app (feature names + types)
    metadata = {
        "target_col": target_col,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "categorical_values": {c: sorted(df[c].dropna().unique().tolist()) for c in categorical_cols},
        "best_params": {k: str(v) for k, v in grid_search.best_params_.items()},
        "cv_mean_roc_auc": float(cv_scores.mean()),
        "cv_std_roc_auc": float(cv_scores.std()),
        "test_metrics": results_df[results_df["model"] == "AdaBoost (tuned)"].to_dict("records")[0],
        "best_overall_model_on_test": best_model_name,
    }
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print("\nAll outputs written to:", OUTPUT_DIR)
    print("Plots written to:", PLOTS_DIR)
    print("Model + metadata written to:", MODELS_DIR)
    return results_df, metadata


if __name__ == "__main__":
    main()
