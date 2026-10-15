"""Data loader, schema validator, and dataset summarization utilities."""

from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
from src.config import PROJECT_ROOT


def load_raw(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Load raw dataset from the path specified in config."""
    rel_path = cfg["dataset"]["path"]
    csv_path = Path(rel_path)
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw dataset file not found at: {csv_path}."
        )

    delimiter = cfg["dataset"].get("delimiter", ";")
    try:
        return pd.read_csv(csv_path, sep=delimiter)
    except Exception:
        # Fallback to automatic separator detection
        return pd.read_csv(csv_path, sep=None, engine="python")


def validate_schema(
    df: pd.DataFrame, cfg: Dict[str, Any], require_target: bool = True
) -> List[str]:
    """Validate DataFrame against the schema defined in config.

    Args:
        df: Input DataFrame.
        cfg: Configuration dictionary.
        require_target: Whether target_col must be present.

    Returns:
        List[str]: List of error messages. Empty list indicates valid schema.
    """
    errors: List[str] = []
    dataset_cfg = cfg["dataset"]
    mode = dataset_cfg.get("mode", "single_record")

    # If raw_features is specified, check against raw features
    raw_features = dataset_cfg.get("raw_features")
    if raw_features:
        expected_cols = list(raw_features)
    else:
        expected_cols = []
        if mode == "longitudinal":
            expected_cols.append(dataset_cfg.get("patient_id_col", "patient_id"))
            expected_cols.append(dataset_cfg.get("date_col", "date"))
        expected_cols.extend(dataset_cfg.get("static_numeric", []))
        expected_cols.extend(dataset_cfg.get("static_categorical", []))
        expected_cols.extend(dataset_cfg.get("time_varying_numeric", []))

    if require_target and dataset_cfg.get("target_col"):
        expected_cols.append(dataset_cfg["target_col"])

    # Check missing columns
    for col in expected_cols:
        if col not in df.columns:
            errors.append(f"Missing required column: '{col}'")

    if errors:
        return errors

    # Check date column parseability if in longitudinal mode
    if mode == "longitudinal" and "date_col" in dataset_cfg:
        date_col = dataset_cfg["date_col"]
        try:
            pd.to_datetime(df[date_col].dropna().head(20))
        except Exception:
            errors.append(f"Date column '{date_col}' contains unparseable dates.")

    # Check numeric columns have numeric-convertible values
    num_cols = [c for c in ["height", "weight", "ap_hi", "ap_lo", "age"] if c in df.columns]
    for col in num_cols:
        non_null = df[col].dropna()
        if len(non_null) > 0:
            converted = pd.to_numeric(non_null, errors="coerce")
            if converted.isna().sum() > len(non_null) * 0.5:
                errors.append(f"Column '{col}' is expected to be numeric but cannot be parsed.")

    return errors


def get_upload_template(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Generate a sample 2-row template DataFrame for user uploads."""
    dataset_cfg = cfg["dataset"]
    mode = dataset_cfg.get("mode", "single_record")

    if mode == "longitudinal":
        pid_col = dataset_cfg.get("patient_id_col", "patient_id")
        date_col = dataset_cfg.get("date_col", "date")
        row1 = {
            pid_col: "P0001",
            date_col: "2021-01-15",
            "age": 45,
            "gender": "F",
            "bmi": 26.5,
            "systolic_bp": 124.0,
            "glucose": 98.0,
            "cholesterol": 190.0,
            "medication_count": 1,
        }
        row2 = {
            pid_col: "P0001",
            date_col: "2021-03-20",
            "age": 45,
            "gender": "F",
            "bmi": 27.0,
            "systolic_bp": 130.0,
            "glucose": 104.0,
            "cholesterol": 195.0,
            "medication_count": 1,
        }
    else:
        # Cardio train template
        row1 = {
            "age": 50,           # Age in years (or days e.g. 18250)
            "gender": 1,         # 1: Female, 2: Male
            "height": 165,       # Height in cm
            "weight": 68.0,      # Weight in kg
            "ap_hi": 120,        # Systolic blood pressure (mmHg)
            "ap_lo": 80,         # Diastolic blood pressure (mmHg)
            "cholesterol": 1,    # 1: Normal, 2: Above normal, 3: Well above normal
            "gluc": 1,           # 1: Normal, 2: Above normal, 3: Well above normal
            "smoke": 0,          # 0: Non-smoker, 1: Smoker
            "alco": 0,           # 0: Non-drinker, 1: Drinker
            "active": 1,         # 0: Not physically active, 1: Physically active
        }
        row2 = {
            "age": 55,
            "gender": 2,
            "height": 178,
            "weight": 86.0,
            "ap_hi": 145,
            "ap_lo": 95,
            "cholesterol": 2,
            "gluc": 2,
            "smoke": 1,
            "alco": 1,
            "active": 0,
        }

    return pd.DataFrame([row1, row2])


def dataset_summary(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Compute comprehensive dataset statistics for reporting and UI."""
    dataset_cfg = cfg["dataset"]
    mode = dataset_cfg.get("mode", "single_record")
    target_col = dataset_cfg.get("target_col")

    n_rows = len(df)
    n_cols = len(df.columns)
    missing_counts = df.isna().sum().to_dict()

    summary: Dict[str, Any] = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "missing_counts": missing_counts,
        "mode": mode,
    }

    if mode == "longitudinal":
        pid_col = dataset_cfg.get("patient_id_col", "patient_id")
        date_col = dataset_cfg.get("date_col", "date")
        if pid_col in df.columns:
            n_patients = df[pid_col].nunique()
            counts_per_patient = df.groupby(pid_col).size()
            min_recs = int(counts_per_patient.min()) if not counts_per_patient.empty else 0
            max_recs = int(counts_per_patient.max()) if not counts_per_patient.empty else 0
            mean_recs = round(float(counts_per_patient.mean()), 1) if not counts_per_patient.empty else 0.0

            dates = pd.to_datetime(df[date_col], errors="coerce").dropna() if date_col in df.columns else pd.Series()
            date_range = (
                f"{dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}"
                if not dates.empty else "N/A"
            )

            summary.update({
                "n_patients": n_patients,
                "records_per_patient": {
                    "min": min_recs,
                    "max": max_recs,
                    "mean": mean_recs,
                },
                "date_range": date_range,
            })
    else:
        summary["n_patients"] = n_rows

    if target_col and target_col in df.columns:
        counts = df[target_col].value_counts().to_dict()
        total = float(len(df[target_col].dropna()))
        proportions = {str(k): round(v / total, 4) for k, v in counts.items()} if total > 0 else {}
        summary["class_balance"] = proportions

    return summary
