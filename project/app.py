"""
app.py
---------------------------------------------------------------
Flask web interface for the Pancreatic Cancer Detection project.

Loads the final trained pipeline (preprocessing + feature selection +
tuned AdaBoost classifier) and model_metadata.json, dynamically builds
an input form for every feature the model expects, and returns a
prediction + probability + SHAP-based per-patient explanation.

*** RESEARCH / DECISION-SUPPORT PROTOTYPE ONLY ***
This is NOT a medical device and must NOT be used for real clinical
diagnosis. It is trained on SYNTHETIC data for demonstration purposes.
---------------------------------------------------------------
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "final_adaboost_pipeline.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"

app = Flask(__name__)

# Load model + metadata once at startup
pipeline = joblib.load(MODEL_PATH)
with open(METADATA_PATH) as f:
    metadata = json.load(f)

NUMERIC_COLS = metadata["numeric_cols"]
CATEGORICAL_COLS = metadata["categorical_cols"]
CATEGORICAL_VALUES = metadata["categorical_values"]
TEST_METRICS = metadata["test_metrics"]


def build_input_dataframe(form_data: dict) -> pd.DataFrame:
    """Turn Flask form data into a single-row DataFrame matching training schema."""
    row = {}
    for col in NUMERIC_COLS:
        val = form_data.get(col, "").strip()
        row[col] = float(val) if val != "" else np.nan
    for col in CATEGORICAL_COLS:
        val = form_data.get(col, "").strip()
        row[col] = val if val != "" else np.nan
    return pd.DataFrame([row])


def explain_prediction(input_df: pd.DataFrame):
    """Return top contributing features (SHAP) for this single prediction."""
    pre = pipeline.named_steps["preprocessor"]
    fsel = pipeline.named_steps["feature_select"]
    clf = pipeline.named_steps["classifier"]

    num_feature_names = NUMERIC_COLS
    ohe = pre.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = list(ohe.get_feature_names_out(CATEGORICAL_COLS)) if CATEGORICAL_COLS else []
    all_feature_names = np.array(num_feature_names + cat_feature_names)

    X_t = pre.transform(input_df)
    if hasattr(X_t, "toarray"):
        X_t = X_t.toarray()
    X_sel = fsel.transform(X_t)
    selected_names = all_feature_names[fsel.get_support()]
    X_sel_df = pd.DataFrame(X_sel, columns=selected_names)

    # Lightweight background for the explainer (zeros = "average" scaled patient)
    background = pd.DataFrame(np.zeros((1, X_sel_df.shape[1])), columns=selected_names)
    explainer = shap.Explainer(clf.predict_proba, background)
    sv = explainer(X_sel_df)

    values = sv.values[0, :, 1] if sv.values.ndim == 3 else sv.values[0]
    contributions = sorted(zip(selected_names, values), key=lambda x: abs(x[1]), reverse=True)
    return contributions[:6]


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    contributions = None
    error = None

    if request.method == "POST":
        try:
            input_df = build_input_dataframe(request.form)
            proba = pipeline.predict_proba(input_df)[0, 1]
            pred = int(proba >= 0.5)
            prediction = "Cancer-suspicious (positive)" if pred == 1 else "Non-cancer (negative)"
            probability = round(float(proba) * 100, 2)
            contributions = explain_prediction(input_df)
        except Exception as e:
            error = f"Could not compute a prediction: {e}"

    return render_template(
        "index.html",
        numeric_cols=NUMERIC_COLS,
        categorical_cols=CATEGORICAL_COLS,
        categorical_values=CATEGORICAL_VALUES,
        prediction=prediction,
        probability=probability,
        contributions=contributions,
        error=error,
        test_metrics=TEST_METRICS,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
