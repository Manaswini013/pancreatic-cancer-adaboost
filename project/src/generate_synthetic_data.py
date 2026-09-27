"""
generate_synthetic_data.py
---------------------------------------------------------------
Generates a SYNTHETIC clinical dataset for the pancreatic cancer
detection project.

IMPORTANT / DISCLAIMER
This data is NOT real patient data. It is artificially generated
using randomised distributions loosely inspired by publicly known
ranges for urinary-biomarker pancreatic cancer studies (e.g. age,
sex ratio, creatinine, LYVE1, REG1B, TFF1, CA19-9). It is created
purely so that the ML pipeline in this project (preprocessing,
feature selection, imbalance handling, CV, tuning, evaluation,
SHAP explainability) can be demonstrated end-to-end.

DO NOT use this file, or any model trained on it, for real
diagnostic, clinical, or medical decision-making of any kind.
---------------------------------------------------------------
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_PATIENTS = 1200          # total synthetic patients
POSITIVE_RATE = 0.18       # ~18% cancer positive -> realistic class imbalance
MISSING_RATE_LOW = 0.02    # light missingness for core biomarkers
MISSING_RATE_HIGH = 0.35   # heavy missingness for an optional biomarker


def generate(n_patients: int = N_PATIENTS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    n_pos = int(n_patients * POSITIVE_RATE)
    n_neg = n_patients - n_pos
    target = np.array([1] * n_pos + [0] * n_neg)
    rng.shuffle(target)

    df = pd.DataFrame({"patient_id": [f"P{str(i).zfill(5)}" for i in range(1, n_patients + 1)]})
    df["target"] = target  # 1 = pancreatic cancer (PDAC), 0 = non-cancer (healthy/benign)

    # ---- Demographics -------------------------------------------------
    df["age"] = rng.normal(loc=np.where(df["target"] == 1, 66, 58), scale=10).round(1)
    df["age"] = df["age"].clip(20, 95)

    df["sex"] = rng.choice(["M", "F"], size=n_patients, p=[0.51, 0.49])

    # sample origin / cohort -> categorical clinical metadata
    df["sample_origin"] = rng.choice(["LIV", "UCL", "SPA", "LON"], size=n_patients,
                                      p=[0.30, 0.30, 0.20, 0.20])
    df["patient_cohort"] = rng.choice(["Cohort1", "Cohort2"], size=n_patients, p=[0.55, 0.45])

    # ---- Urinary biomarkers -------------------------------------------
    # Values are log-normal-ish; cancer patients skew higher on average
    def biomarker(mu_pos, sigma_pos, mu_neg, sigma_neg):
        mu = np.where(df["target"] == 1, mu_pos, mu_neg)
        sigma = np.where(df["target"] == 1, sigma_pos, sigma_neg)
        return np.exp(rng.normal(mu, sigma))

    df["creatinine"] = biomarker(mu_pos=0.55, sigma_pos=0.55, mu_neg=0.35, sigma_neg=0.55)
    df["LYVE1"] = biomarker(mu_pos=1.7, sigma_pos=0.65, mu_neg=0.9, sigma_neg=0.70)
    df["REG1B"] = biomarker(mu_pos=3.6, sigma_pos=0.75, mu_neg=2.4, sigma_neg=0.80)
    df["TFF1"] = biomarker(mu_pos=4.4, sigma_pos=0.85, mu_neg=3.2, sigma_neg=0.90)

    # Optional biomarker measured only in a subset of patients (heavy missingness)
    df["plasma_CA19_9"] = biomarker(mu_pos=4.0, sigma_pos=1.4, mu_neg=2.0, sigma_neg=1.1)

    # REG1A: correlated with REG1B, collected in fewer patients as well
    df["REG1A"] = df["REG1B"] * rng.normal(1.0, 0.15, size=n_patients) + rng.normal(0, 0.5, size=n_patients)
    df["REG1A"] = df["REG1A"].clip(lower=0.01)

    # ---- Inject missing values (realistic pattern) ---------------------
    for col, rate in [("creatinine", MISSING_RATE_LOW),
                      ("LYVE1", MISSING_RATE_LOW),
                      ("REG1B", MISSING_RATE_LOW),
                      ("TFF1", MISSING_RATE_LOW),
                      ("age", 0.01)]:
        mask = rng.random(n_patients) < rate
        df.loc[mask, col] = np.nan

    # Optional biomarkers: missing-not-completely-at-random (collected less for
    # certain cohorts) -> deliberately heavier missingness to exercise imputation
    mask_ca19 = rng.random(n_patients) < MISSING_RATE_HIGH
    df.loc[mask_ca19, "plasma_CA19_9"] = np.nan
    mask_reg1a = rng.random(n_patients) < (MISSING_RATE_HIGH + 0.15)
    df.loc[mask_reg1a, "REG1A"] = np.nan

    # ---- Round numeric biomarker columns for readability ----------------
    for col in ["creatinine", "LYVE1", "REG1B", "TFF1", "plasma_CA19_9", "REG1A"]:
        df[col] = df[col].round(3)

    # Shuffle row order
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return df


if __name__ == "__main__":
    data = generate()
    out_path = "/home/claude/project/data/synthetic_pancreatic_clinical_data.csv"
    data.to_csv(out_path, index=False)
    print(f"Saved synthetic dataset: {out_path}")
    print(data.shape)
    print(data.head())
    print("\nClass balance:\n", data["target"].value_counts(normalize=True))
    print("\nMissing values per column:\n", data.isna().sum())
