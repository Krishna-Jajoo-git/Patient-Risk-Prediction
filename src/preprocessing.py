"""Data cleaning and scikit-learn ColumnTransformer preprocessor construction."""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def clean_data(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean dataset according to validation rules without data leakage.

    Steps:
    1. Parse dates, strip whitespace, and standardize categorical casing.
    2. Drop exact duplicate rows.
    3. Set values outside valid_ranges to NaN (impossible-value handling).
    4. Compute an IQR outlier report per numeric column (count only; no silent row deletion).

    Args:
        df: Input raw DataFrame.
        cfg: Configuration dictionary.

    Returns:
        tuple[pd.DataFrame, dict]: Cleaned DataFrame and comprehensive cleaning report.
    """
    cleaned = df.copy()
    dataset_cfg = cfg["dataset"]
    mode = dataset_cfg["mode"]
    valid_ranges = dataset_cfg.get("valid_ranges", {})

    report: Dict[str, Any] = {
        "initial_rows": len(cleaned),
        "initial_missing": cleaned.isna().sum().to_dict(),
        "duplicates_removed": 0,
        "impossible_values_nulled": {},
        "iqr_outliers": {},
    }

    # Step 1: Parse dates & standardise text
    if mode == "longitudinal":
        date_col = dataset_cfg["date_col"]
        if date_col in cleaned.columns:
            cleaned[date_col] = pd.to_datetime(cleaned[date_col], errors="coerce")

    cat_cols = dataset_cfg.get("static_categorical", [])
    gender_synonyms = {
        "m": "M",
        "male": "M",
        "f": "F",
        "female": "F",
        "other": "Other",
    }
    for col in cat_cols:
        if col in cleaned.columns:
            if col.lower() == "gender":
                cleaned[col] = cleaned[col].apply(
                    lambda x: gender_synonyms.get(str(x).strip().lower(), str(x).strip().capitalize())
                    if pd.notna(x)
                    else np.nan
                )
            else:
                cleaned[col] = cleaned[col].apply(
                    lambda x: str(x).strip().capitalize() if pd.notna(x) else np.nan
                )

    # Step 2: Drop exact duplicates
    dup_count = int(cleaned.duplicated().sum())
    if dup_count > 0:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = dup_count

    # Step 3: Validate ranges (impossible-value handling -> NaN)
    for col, (low, high) in valid_ranges.items():
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
            invalid_mask = (cleaned[col] < low) | (cleaned[col] > high)
            num_invalid = int(invalid_mask.sum())
            report["impossible_values_nulled"][col] = num_invalid
            if num_invalid > 0:
                cleaned.loc[invalid_mask, col] = np.nan

    # Step 4: IQR outlier report per numeric column (flagging only)
    num_cols = dataset_cfg.get("static_numeric", []) + dataset_cfg.get("time_varying_numeric", [])
    for col in num_cols:
        if col in cleaned.columns:
            series = cleaned[col].dropna()
            if len(series) > 4:
                q25 = series.quantile(0.25)
                q75 = series.quantile(0.75)
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                outliers = int(((series < lower_bound) | (series > upper_bound)).sum())
                report["iqr_outliers"][col] = {
                    "count": outliers,
                    "bounds": [round(float(lower_bound), 2), round(float(upper_bound), 2)],
                }

    report["final_rows"] = len(cleaned)
    report["final_missing"] = cleaned.isna().sum().to_dict()

    return cleaned, report


def build_preprocessor(
    cfg: Dict[str, Any], numeric_cols: List[str], categorical_cols: List[str]
) -> ColumnTransformer:
    """Construct an unfitted ColumnTransformer for scikit-learn Pipeline.

    Numeric pipeline:
      - SimpleImputer(strategy='median')
      - StandardScaler()

    Categorical pipeline:
      - SimpleImputer(strategy='most_frequent')
      - OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    Imputation and scaling parameters are strictly learned from training folds only.
    """
    transformers = []

    if numeric_cols:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipeline, numeric_cols))

    if categorical_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipeline, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def plot_missing_before_after(report: Dict[str, Any], output_path: str = None) -> plt.Figure:
    """Generate and save a bar chart comparing missingness before and after cleaning."""
    init_miss = report.get("initial_missing", {})
    final_miss = report.get("final_missing", {})

    cols = sorted(list(init_miss.keys()))
    if not cols:
        return None

    x = np.arange(len(cols))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, [init_miss[c] for c in cols], width, label="Before Cleaning (Raw)", color="#ef4444", alpha=0.85)
    ax.bar(x + width / 2, [final_miss.get(c, 0) for c in cols], width, label="After Cleaning (Nulled)", color="#3b82f6", alpha=0.85)

    ax.set_ylabel("Count of Missing / Invalid Values")
    ax.set_title("Missing and Invalid Value Profile: Before vs. After Cleaning")
    ax.set_xticks(x)
    ax.set_xticklabels(cols, rotation=35, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=150)
        plt.close(fig)

    return fig
