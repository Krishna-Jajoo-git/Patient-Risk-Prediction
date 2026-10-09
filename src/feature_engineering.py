"""Longitudinal feature aggregation and patient-level feature engineering.

Improvements:
- Added clinical domain features: BMI category flags, hypertension flags,
  diabetes-risk flags, composite risk score.
- Added missingness indicator features (missingness is often informative).
- Added trend-slope feature using linear regression over visits for vitals.
- Cross-feature interactions (e.g., glucose × bmi, age × bp).
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd


def get_feature_lists(cfg: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Return lists of engineered numeric and categorical feature names."""
    dataset_cfg = cfg["dataset"]
    mode = dataset_cfg["mode"]

    if mode == "single_record":
        num_cols = list(dataset_cfg.get("static_numeric", [])) + list(
            dataset_cfg.get("time_varying_numeric", [])
        )
        cat_cols = list(dataset_cfg.get("static_categorical", []))
        return num_cols, cat_cols

    # Longitudinal engineered features
    num_cols = []
    tv_cols = dataset_cfg.get("time_varying_numeric", [])
    for col in tv_cols:
        num_cols.extend([
            f"{col}_latest",
            f"{col}_first",
            f"{col}_mean",
            f"{col}_min",
            f"{col}_max",
            f"{col}_std",
            f"{col}_change",
            f"{col}_pct_change",
            f"{col}_trend_slope",     # NEW: linear trend slope over visits
            f"{col}_missing_rate",    # NEW: fraction of records where this was missing
        ])

    # Global trajectory features
    num_cols.extend(["n_records", "span_days", "mean_gap_days"])

    # Static features
    num_cols.extend(dataset_cfg.get("static_numeric", []))

    # --- Domain-specific derived features ---
    # These are computed if the corresponding raw columns are present
    domain_numeric = [
        "composite_risk_score",     # weighted clinical risk index
        "bmi_obese",                # BMI >= 30
        "bmi_overweight",           # 25 <= BMI < 30
        "hypertension_flag",        # systolic_bp_latest >= 140
        "diabetes_risk_flag",       # glucose_latest >= 126 or glucose_mean >= 100
        "high_cholesterol_flag",    # cholesterol_latest >= 200
        "age_x_glucose",            # interaction: age × mean glucose
        "age_x_bp",                 # interaction: age × mean systolic_bp
        "bmi_x_glucose",            # interaction: bmi × mean glucose
        "bp_glucose_product",       # systolic_bp_latest × glucose_latest
        "poly_age2",                # age^2 to capture non-linear age effects
    ]
    num_cols.extend(domain_numeric)

    cat_cols = list(dataset_cfg.get("static_categorical", []))

    return num_cols, cat_cols


def _linear_slope(series: pd.Series) -> float:
    """Compute slope of best-fit line through (index, value) pairs. Returns 0 for < 2 points."""
    vals = series.dropna().values
    if len(vals) < 2:
        return 0.0
    x = np.arange(len(vals), dtype=float)
    # Simple closed-form least-squares slope
    x_mean = x.mean()
    y_mean = vals.mean()
    ss_x = ((x - x_mean) ** 2).sum()
    if ss_x == 0:
        return 0.0
    slope = ((x - x_mean) * (vals - y_mean)).sum() / ss_x
    return float(slope)


def build_patient_features(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Aggregate longitudinal records into one feature row per patient.

    In single_record mode, returns the selected columns directly.
    In longitudinal mode, sorts records chronologically per patient and computes:
      - Latest, first, mean, min, max, std, absolute change, percent change for vitals.
      - Trend slope (linear regression over visits) for vitals.
      - Missingness rate for each vital per patient.
      - Total records (n_records), time span in days (span_days), average visit gap.
      - Static baseline features and target label.
      - Clinical domain features: BMI flags, hypertension, diabetes risk, interactions.
    """
    dataset_cfg = cfg["dataset"]
    mode = dataset_cfg["mode"]
    target_col = dataset_cfg.get("target_col")

    if mode == "single_record":
        num_cols, cat_cols = get_feature_lists(cfg)
        cols_to_keep = num_cols + cat_cols
        if target_col and target_col in df.columns:
            cols_to_keep.append(target_col)
        return df[cols_to_keep].copy()

    pid_col = dataset_cfg["patient_id_col"]
    date_col = dataset_cfg["date_col"]
    tv_cols = dataset_cfg.get("time_varying_numeric", [])
    static_num = dataset_cfg.get("static_numeric", [])
    static_cat = dataset_cfg.get("static_categorical", [])

    # Sort by patient and chronological date
    work_df = df.copy()
    work_df[date_col] = pd.to_datetime(work_df[date_col], errors="coerce")
    work_df = work_df.sort_values(by=[pid_col, date_col]).reset_index(drop=True)

    patient_rows = []

    for pid, group in work_df.groupby(pid_col, sort=False):
        row: Dict[str, Any] = {pid_col: pid}
        n_recs = len(group)
        row["n_records"] = n_recs

        # Date calculations
        valid_dates = group[date_col].dropna()
        if len(valid_dates) >= 2:
            span = (valid_dates.iloc[-1] - valid_dates.iloc[0]).total_seconds() / 86400.0
            gaps = valid_dates.diff().dt.total_seconds().dropna() / 86400.0
            mean_gap = float(gaps.mean()) if len(gaps) > 0 else 0.0
        else:
            span = 0.0
            mean_gap = 0.0

        row["span_days"] = round(float(span), 2)
        row["mean_gap_days"] = round(float(mean_gap), 2)

        # Time-varying vital aggregations
        vital_aggs: Dict[str, Dict] = {}
        for col in tv_cols:
            raw_series = group[col]
            vals = pd.to_numeric(raw_series, errors="coerce").dropna()

            missing_rate = round(float(raw_series.isna().sum()) / n_recs, 4)
            row[f"{col}_missing_rate"] = missing_rate

            if len(vals) == 0:
                row[f"{col}_latest"] = np.nan
                row[f"{col}_first"] = np.nan
                row[f"{col}_mean"] = np.nan
                row[f"{col}_min"] = np.nan
                row[f"{col}_max"] = np.nan
                row[f"{col}_std"] = 0.0
                row[f"{col}_change"] = 0.0
                row[f"{col}_pct_change"] = np.nan
                row[f"{col}_trend_slope"] = 0.0
                vital_aggs[col] = {"mean": np.nan, "latest": np.nan}
            else:
                first_v = float(vals.iloc[0])
                latest_v = float(vals.iloc[-1])
                mean_v = float(vals.mean())
                min_v = float(vals.min())
                max_v = float(vals.max())
                std_v = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                if np.isnan(std_v):
                    std_v = 0.0

                chg = latest_v - first_v
                pct_chg = (chg / first_v) if first_v != 0 else np.nan
                slope = _linear_slope(vals)

                row[f"{col}_latest"] = round(latest_v, 2)
                row[f"{col}_first"] = round(first_v, 2)
                row[f"{col}_mean"] = round(mean_v, 2)
                row[f"{col}_min"] = round(min_v, 2)
                row[f"{col}_max"] = round(max_v, 2)
                row[f"{col}_std"] = round(std_v, 2)
                row[f"{col}_change"] = round(chg, 2)
                row[f"{col}_pct_change"] = round(float(pct_chg), 4) if not np.isnan(pct_chg) else np.nan
                row[f"{col}_trend_slope"] = round(slope, 4)
                vital_aggs[col] = {"mean": mean_v, "latest": latest_v}

        # Static attributes
        age_val = np.nan
        for col in static_num:
            vals = pd.to_numeric(group[col], errors="coerce").dropna()
            val = float(vals.iloc[-1]) if len(vals) > 0 else np.nan
            row[col] = val
            if col == "age":
                age_val = val

        for col in static_cat:
            vals = group[col].dropna()
            row[col] = vals.iloc[-1] if len(vals) > 0 else np.nan

        # --- Clinical domain features ---
        bmi_mean = vital_aggs.get("bmi", {}).get("mean", np.nan)
        bmi_latest = vital_aggs.get("bmi", {}).get("latest", np.nan)
        bp_latest = vital_aggs.get("systolic_bp", {}).get("latest", np.nan)
        bp_mean = vital_aggs.get("systolic_bp", {}).get("mean", np.nan)
        gluc_latest = vital_aggs.get("glucose", {}).get("latest", np.nan)
        gluc_mean = vital_aggs.get("glucose", {}).get("mean", np.nan)
        chol_latest = vital_aggs.get("cholesterol", {}).get("latest", np.nan)

        # BMI category flags (0/1)
        row["bmi_obese"] = int(bmi_latest >= 30) if not np.isnan(bmi_latest) else 0
        row["bmi_overweight"] = int(25 <= bmi_latest < 30) if not np.isnan(bmi_latest) else 0

        # Hypertension (systolic >= 140)
        row["hypertension_flag"] = int(bp_latest >= 140) if not np.isnan(bp_latest) else 0

        # Diabetes risk (fasting glucose >= 126 mg/dL OR mean >= 100)
        diab_flag = 0
        if not np.isnan(gluc_latest) and gluc_latest >= 126:
            diab_flag = 1
        elif not np.isnan(gluc_mean) and gluc_mean >= 100:
            diab_flag = 1
        row["diabetes_risk_flag"] = diab_flag

        # High cholesterol (>= 200 mg/dL)
        row["high_cholesterol_flag"] = int(chol_latest >= 200) if not np.isnan(chol_latest) else 0

        # Composite risk score (weighted sum of key risk factors, normalized)
        # Weights based on clinical significance
        risk = 0.0
        if not np.isnan(age_val):
            risk += (age_val / 100.0) * 2.0
        if not np.isnan(bmi_mean):
            risk += max(0.0, (bmi_mean - 18.5) / 40.0) * 1.5
        if not np.isnan(bp_mean):
            risk += max(0.0, (bp_mean - 90.0) / 160.0) * 2.0
        if not np.isnan(gluc_mean):
            risk += max(0.0, (gluc_mean - 70.0) / 200.0) * 2.0
        if not np.isnan(chol_latest):
            risk += max(0.0, (chol_latest - 150.0) / 300.0) * 1.0
        row["composite_risk_score"] = round(risk, 4)

        # Interaction features
        row["age_x_glucose"] = round(age_val * gluc_mean, 2) if not (np.isnan(age_val) or np.isnan(gluc_mean)) else np.nan
        row["age_x_bp"] = round(age_val * bp_mean, 2) if not (np.isnan(age_val) or np.isnan(bp_mean)) else np.nan
        row["bmi_x_glucose"] = round(bmi_mean * gluc_mean, 2) if not (np.isnan(bmi_mean) or np.isnan(gluc_mean)) else np.nan
        row["bp_glucose_product"] = round(bp_latest * gluc_latest, 2) if not (np.isnan(bp_latest) or np.isnan(gluc_latest)) else np.nan
        row["poly_age2"] = round(age_val ** 2, 2) if not np.isnan(age_val) else np.nan

        # Target outcome (last recorded)
        if target_col and target_col in group.columns:
            t_vals = group[target_col].dropna()
            row[target_col] = t_vals.iloc[-1] if len(t_vals) > 0 else np.nan

        patient_rows.append(row)

    return pd.DataFrame(patient_rows)


def summarize_patient(patient_df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Compute summary KPIs for display cards in the user interface."""
    dataset_cfg = cfg["dataset"]
    mode = dataset_cfg["mode"]
    tv_cols = dataset_cfg.get("time_varying_numeric", [])

    summary: Dict[str, Any] = {
        "n_records": len(patient_df),
        "vitals": {},
    }

    if mode == "longitudinal":
        date_col = dataset_cfg["date_col"]
        valid_dates = pd.to_datetime(patient_df[date_col], errors="coerce").dropna()
        if len(valid_dates) >= 2:
            span_days = int((valid_dates.iloc[-1] - valid_dates.iloc[0]).total_seconds() / 86400.0)
            summary["span_days"] = span_days
            summary["date_range"] = f"{valid_dates.iloc[0].strftime('%b %d, %Y')} - {valid_dates.iloc[-1].strftime('%b %d, %Y')}"
        else:
            summary["span_days"] = 0
            summary["date_range"] = "Single observation"

    for col in tv_cols:
        if col in patient_df.columns:
            series = pd.to_numeric(patient_df[col], errors="coerce").dropna()
            if len(series) > 0:
                first_val = float(series.iloc[0])
                latest_val = float(series.iloc[-1])
                mean_val = float(series.mean())
                chg = latest_val - first_val
                pct = ((chg / first_val) * 100) if first_val != 0 else 0.0
                summary["vitals"][col] = {
                    "first": round(first_val, 1),
                    "latest": round(latest_val, 1),
                    "mean": round(mean_val, 1),
                    "change": round(chg, 1),
                    "pct_change": round(pct, 1),
                }

    return summary
