"""Feature engineering for Cardiovascular Disease dataset (cardio_train) and single-record clinical evaluation.

Engineered features:
- age_years: Standardized age in calendar years (computed from days or years).
- bmi: Body Mass Index (weight_kg / (height_m)^2).
- bmi_category: Underweight, Normal, Overweight, Obese.
- pulse_pressure: Systolic BP minus Diastolic BP (marker of arterial stiffness).
- map: Mean Arterial Pressure (ap_lo + (ap_hi - ap_lo) / 3).
- bp_category: AHA/ACC clinical blood pressure category (Normal, Elevated, Stage 1 HTN, Stage 2 HTN, Crisis).
- age_x_bp: Non-linear interaction between age and systolic blood pressure.
- composite_risk_score: Multivariable cardiovascular risk score incorporating hemodynamics, metabolism, and lifestyle.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd


def get_feature_lists(cfg: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Return lists of engineered numeric and categorical feature names."""
    dataset_cfg = cfg["dataset"]
    mode = dataset_cfg.get("mode", "single_record")

    num_cols = list(dataset_cfg.get("static_numeric", [
        "age_years",
        "height",
        "weight",
        "ap_hi",
        "ap_lo",
        "bmi",
        "pulse_pressure",
        "map",
        "age_x_bp",
        "composite_risk_score",
    ]))

    cat_cols = list(dataset_cfg.get("static_categorical", [
        "gender",
        "cholesterol",
        "gluc",
        "smoke",
        "alco",
        "active",
        "bp_category",
        "bmi_category",
    ]))

    return num_cols, cat_cols


def _compute_bp_category(ap_hi: float, ap_lo: float) -> str:
    """Classify blood pressure into AHA/ACC guideline stages."""
    if np.isnan(ap_hi) or np.isnan(ap_lo):
        return "Unknown"
    if ap_hi >= 180 or ap_lo >= 120:
        return "Crisis"
    if ap_hi >= 140 or ap_lo >= 90:
        return "Stage 2 HTN"
    if ap_hi >= 130 or ap_lo >= 80:
        return "Stage 1 HTN"
    if ap_hi >= 120 and ap_lo < 80:
        return "Elevated"
    if ap_hi < 120 and ap_lo < 80:
        return "Normal"
    return "Stage 1 HTN"


def _compute_bmi_category(bmi: float) -> str:
    """Classify Body Mass Index into WHO weight categories."""
    if np.isnan(bmi):
        return "Unknown"
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25.0:
        return "Normal"
    if bmi < 30.0:
        return "Overweight"
    return "Obese"


def build_patient_features(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Transform raw cardiovascular patient records into engineered clinical feature representations.

    Supports both batch datasets and individual patient inputs.
    """
    dataset_cfg = cfg["dataset"]
    target_col = dataset_cfg.get("target_col", "cardio")
    work = df.copy()

    # 1. Age in years: if age > 120, raw value is in days (as in cardio_train.csv)
    if "age_years" not in work.columns and "age" in work.columns:
        work["age"] = pd.to_numeric(work["age"], errors="coerce")
        work["age_years"] = np.where(work["age"] > 120, work["age"] / 365.25, work["age"])
    elif "age_years" not in work.columns and "age" not in work.columns:
        work["age_years"] = np.nan

    # 2. Hemodynamics & Anthropometrics
    for col in ["height", "weight", "ap_hi", "ap_lo"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    # Body Mass Index (BMI)
    if "bmi" not in work.columns:
        if "height" in work.columns and "weight" in work.columns:
            height_m = work["height"] / 100.0
            work["bmi"] = work["weight"] / (height_m ** 2)
            # Clip extreme nonsensical BMI values to valid bounds
            work.loc[(work["bmi"] < 10) | (work["bmi"] > 70), "bmi"] = np.nan
        else:
            work["bmi"] = np.nan

    # Pulse Pressure: ap_hi - ap_lo
    if "pulse_pressure" not in work.columns:
        if "ap_hi" in work.columns and "ap_lo" in work.columns:
            work["pulse_pressure"] = work["ap_hi"] - work["ap_lo"]
        else:
            work["pulse_pressure"] = np.nan

    # Mean Arterial Pressure (MAP): DBP + 1/3 (SBP - DBP)
    if "map" not in work.columns:
        if "ap_hi" in work.columns and "ap_lo" in work.columns:
            work["map"] = work["ap_lo"] + (work["ap_hi"] - work["ap_lo"]) / 3.0
        else:
            work["map"] = np.nan

    # AHA Blood Pressure Category
    if "bp_category" not in work.columns:
        if "ap_hi" in work.columns and "ap_lo" in work.columns:
            work["bp_category"] = [
                _compute_bp_category(h, l) for h, l in zip(work["ap_hi"], work["ap_lo"])
            ]
        else:
            work["bp_category"] = "Normal"

    # WHO BMI Category
    if "bmi_category" not in work.columns:
        if "bmi" in work.columns:
            work["bmi_category"] = [_compute_bmi_category(b) for b in work["bmi"]]
        else:
            work["bmi_category"] = "Normal"

    # 3. Interactions & Non-linear combinations
    if "age_x_bp" not in work.columns:
        if "age_years" in work.columns and "ap_hi" in work.columns:
            work["age_x_bp"] = (work["age_years"] * (work["ap_hi"] / 100.0)).round(2)
        else:
            work["age_x_bp"] = np.nan

    # 4. Multivariable Composite Cardiovascular Risk Score
    if "composite_risk_score" not in work.columns:
        age_y = work.get("age_years", pd.Series(50, index=work.index)).fillna(50)
        ap_h = work.get("ap_hi", pd.Series(120, index=work.index)).fillna(120)
        chol = work.get("cholesterol", pd.Series(1, index=work.index)).fillna(1).astype(float)
        gluc = work.get("gluc", pd.Series(1, index=work.index)).fillna(1).astype(float)
        smoke = work.get("smoke", pd.Series(0, index=work.index)).fillna(0).astype(float)
        act = work.get("active", pd.Series(1, index=work.index)).fillna(1).astype(float)
        bmi_val = work.get("bmi", pd.Series(24, index=work.index)).fillna(24)

        age_term = np.maximum(0, (age_y - 30) / 35.0) * 2.0
        bp_term = np.maximum(0, (ap_h - 110) / 60.0) * 2.5
        chol_term = (chol - 1.0) * 1.5
        gluc_term = (gluc - 1.0) * 1.2
        smoke_term = smoke * 1.8
        act_term = (1.0 - act) * 0.8
        bmi_term = np.maximum(0, (bmi_val - 22.0) / 15.0) * 1.2

        composite = (age_term + bp_term + chol_term + gluc_term + smoke_term + act_term + bmi_term) / 10.0
        work["composite_risk_score"] = composite.round(4)

    # 5. Categorical column formatting
    for c in ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]:
        if c in work.columns:
            work[c] = work[c].astype(str)

    num_cols, cat_cols = get_feature_lists(cfg)
    cols_to_keep = [c for c in num_cols + cat_cols if c in work.columns]

    if "id" in work.columns:
        cols_to_keep.insert(0, "id")
    if target_col in work.columns:
        cols_to_keep.append(target_col)

    return work[cols_to_keep].copy()


def summarize_patient(patient_df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract clinical summary metrics for a single patient record to render KPI display cards."""
    if patient_df.empty:
        return {}

    row = patient_df.iloc[-1]

    # Age
    raw_age = row.get("age", np.nan)
    if pd.notna(raw_age):
        age_years = round(float(raw_age) / 365.25, 1) if float(raw_age) > 120 else round(float(raw_age), 1)
    else:
        age_years = row.get("age_years", 50.0)

    # Gender
    gender_raw = str(row.get("gender", "1"))
    gender_label = "Female" if gender_raw in ("1", "1.0", "F", "Female") else "Male"

    # Anthropometrics
    height = float(row.get("height", 165))
    weight = float(row.get("weight", 70))
    bmi = float(row.get("bmi", weight / ((height / 100.0) ** 2))) if height > 0 else 24.0

    # Hemodynamics
    ap_hi = float(row.get("ap_hi", 120))
    ap_lo = float(row.get("ap_lo", 80))
    pulse_pressure = ap_hi - ap_lo
    mean_ap = ap_lo + pulse_pressure / 3.0

    # Metabolic & Behavioral
    chol = int(float(row.get("cholesterol", 1)))
    gluc = int(float(row.get("gluc", 1)))
    smoke = int(float(row.get("smoke", 0)))
    alco = int(float(row.get("alco", 0)))
    active = int(float(row.get("active", 1)))

    chol_labels = {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}
    gluc_labels = {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}

    return {
        "demographics": {
            "age_years": age_years,
            "gender": gender_label,
            "height": height,
            "weight": weight,
            "bmi": round(bmi, 1),
            "bmi_category": _compute_bmi_category(bmi),
        },
        "hemodynamics": {
            "ap_hi": round(ap_hi, 0),
            "ap_lo": round(ap_lo, 0),
            "pulse_pressure": round(pulse_pressure, 0),
            "mean_arterial_pressure": round(mean_ap, 1),
            "bp_category": _compute_bp_category(ap_hi, ap_lo),
        },
        "metabolic": {
            "cholesterol_level": chol,
            "cholesterol_label": chol_labels.get(chol, "Normal"),
            "glucose_level": gluc,
            "glucose_label": gluc_labels.get(gluc, "Normal"),
        },
        "lifestyle": {
            "smoking": "Yes (Smoker)" if smoke == 1 else "No (Non-smoker)",
            "alcohol": "Yes (Drinks alcohol)" if alco == 1 else "No (Non-drinker)",
            "physical_activity": "Active" if active == 1 else "Sedentary / Inactive",
        },
    }
