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
            f"Raw dataset file not found at: {csv_path}. "
            "Run 'python -m src.generate_demo_data' if using demo data."
        )

    return pd.read_csv(csv_path)


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
    mode = dataset_cfg["mode"]

    # Required columns
    expected_cols = []
    if mode == "longitudinal":
        expected_cols.append(dataset_cfg["patient_id_col"])
        expected_cols.append(dataset_cfg["date_col"])

    expected_cols.extend(dataset_cfg.get("static_numeric", []))
    expected_cols.extend(dataset_cfg.get("static_categorical", []))
    expected_cols.extend(dataset_cfg.get("time_varying_numeric", []))

    if require_target:
        expected_cols.append(dataset_cfg["target_col"])

    # Check missing columns
    for col in expected_cols:
        if col not in df.columns:
            errors.append(f"Missing required column: '{col}'")

    if errors:
        return errors

    # Check date column parseability
    if mode == "longitudinal":
        date_col = dataset_cfg["date_col"]
        try:
            pd.to_datetime(df[date_col].dropna().head(20))
        except Exception:
            errors.append(f"Date column '{date_col}' contains unparseable dates.")

    # Check numeric columns have numeric-convertible values
    num_cols = dataset_cfg.get("static_numeric", []) + dataset_cfg.get("time_varying_numeric", [])
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
    mode = dataset_cfg["mode"]

    if mode == "longitudinal":
        pid_col = dataset_cfg["patient_id_col"]
        date_col = dataset_cfg["date_col"]
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
        row1 = {
            "age": 52,
            "gender": "M",
            "bmi": 28.2,
            "systolic_bp": 136.0,
            "glucose": 115.0,
            "cholesterol": 210.0,
            "medication_count": 2,
        }
        row2 = {
            "age": 61,
            "gender": "F",
            "bmi": 24.1,
            "systolic_bp": 118.0,
            "glucose": 92.0,
            "cholesterol": 182.0,
            "medication_count": 0,
        }

    return pd.DataFrame([row1, row2])


def dataset_summary(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Compute comprehensive dataset statistics for reporting and UI."""
    dataset_cfg = cfg["dataset"]
    mode = dataset_cfg["mode"]
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
        pid_col = dataset_cfg["patient_id_col"]
        date_col = dataset_cfg["date_col"]
        n_patients = df[pid_col].nunique()
        counts_per_patient = df.groupby(pid_col).size()
        min_recs = int(counts_per_patient.min()) if not counts_per_patient.empty else 0
        max_recs = int(counts_per_patient.max()) if not counts_per_patient.empty else 0
        mean_recs = round(float(counts_per_patient.mean()), 1) if not counts_per_patient.empty else 0.0

        dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
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

    if target_col and target_col in df.columns:
        counts = df[target_col].value_counts().to_dict()
        total = float(len(df[target_col].dropna()))
        proportions = {str(k): round(v / total, 4) for k, v in counts.items()} if total > 0 else {}
        summary["class_balance"] = proportions

    return summary
