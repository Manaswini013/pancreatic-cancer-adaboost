"""
inspect_data.py
---------------------------------------------------------------
Automatically inspects a clinical dataset CSV:
 - infers the target/label column
 - separates numeric vs categorical vs identifier columns
 - reports missing values
 - reports class distribution / imbalance ratio

This module makes NO hardcoded assumptions about column names.
It uses simple, transparent heuristics so the same script would
work on a different clinical CSV with different column names.
---------------------------------------------------------------
"""

import pandas as pd
import numpy as np


ID_LIKE_TOKENS = ("id", "patient", "sample", "uuid", "index")
TARGET_LIKE_TOKENS = ("target", "label", "diagnosis", "class", "outcome", "cancer", "status")


def guess_target_column(df: pd.DataFrame) -> str:
    """Heuristic target-column detector.

    Priority:
      1. A column whose name matches common target-like tokens AND has a
         small number of unique values (looks categorical/binary).
      2. Fallback: the last binary/low-cardinality column in the dataframe.
    """
    candidates = []
    for col in df.columns:
        name = col.lower()
        n_unique = df[col].nunique(dropna=True)
        if any(tok in name for tok in TARGET_LIKE_TOKENS) and n_unique <= 10:
            candidates.append((col, n_unique))

    if candidates:
        # Prefer the most binary-looking candidate (fewest unique values)
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]

    # Fallback: any low-cardinality (<=10 unique) non-id column, take the last one
    fallback = [c for c in df.columns
                if df[c].nunique(dropna=True) <= 10
                and not any(tok in c.lower() for tok in ID_LIKE_TOKENS)]
    if fallback:
        return fallback[-1]

    raise ValueError("Could not automatically identify a target column. "
                      "Please specify it manually.")


def classify_columns(df: pd.DataFrame, target_col: str):
    """Split columns into id-like, numeric feature, and categorical feature groups.

    A column is treated as an identifier only if its name matches an
    id-like token AND it is (near) unique per row -- e.g. 'patient_id' with
    1200 distinct values out of 1200 rows. This avoids misclassifying
    low-cardinality categorical metadata such as 'sample_origin' or
    'patient_cohort' as identifiers just because they share a substring.
    """
    id_cols, numeric_cols, categorical_cols = [], [], []
    n_rows = len(df)

    for col in df.columns:
        if col == target_col:
            continue
        name = col.lower()
        n_unique = df[col].nunique(dropna=True)
        looks_like_id_name = any(tok in name for tok in ID_LIKE_TOKENS)
        is_near_unique = n_unique >= 0.9 * n_rows

        if looks_like_id_name and is_near_unique:
            id_cols.append(col)
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return id_cols, numeric_cols, categorical_cols


def inspect(csv_path: str, target_col: str = None) -> dict:
    df = pd.read_csv(csv_path)

    if target_col is None:
        target_col = guess_target_column(df)

    id_cols, numeric_cols, categorical_cols = classify_columns(df, target_col)

    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    class_counts = df[target_col].value_counts()
    class_pct = (class_counts / len(df) * 100).round(2)

    report = {
        "shape": df.shape,
        "target_col": target_col,
        "id_cols": id_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "missing_counts": missing[missing > 0].to_dict(),
        "missing_pct": missing_pct[missing_pct > 0].to_dict(),
        "class_counts": class_counts.to_dict(),
        "class_pct": class_pct.to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
    return report, df


def print_report(report: dict):
    print("=" * 70)
    print("DATASET INSPECTION REPORT")
    print("=" * 70)
    print(f"Shape (rows, cols): {report['shape']}")
    print(f"\nAuto-detected target column: '{report['target_col']}'")
    print(f"\nID-like columns (excluded from modelling): {report['id_cols']}")
    print(f"\nNumeric feature columns ({len(report['numeric_cols'])}): {report['numeric_cols']}")
    print(f"\nCategorical feature columns ({len(report['categorical_cols'])}): {report['categorical_cols']}")
    print("\nMissing values (column: count, %):")
    if report["missing_counts"]:
        for col, cnt in report["missing_counts"].items():
            print(f"  - {col}: {cnt} missing ({report['missing_pct'][col]}%)")
    else:
        print("  - No missing values detected.")
    print("\nClass distribution (target column):")
    for cls, cnt in report["class_counts"].items():
        print(f"  - class {cls}: {cnt} samples ({report['class_pct'][cls]}%)")
    print("=" * 70)


if __name__ == "__main__":
    CSV_PATH = "/home/claude/project/data/synthetic_pancreatic_clinical_data.csv"
    report, df = inspect(CSV_PATH)
    print_report(report)
