"""Generates synthetic longitudinal patient data for pipeline testing."""

import datetime
import logging
import sys
from pathlib import Path

# Ensure repository root is on sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from src.config import load_config, PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SYNTHETIC_BANNER = (
    "=" * 70 + "\n"
    "SYNTHETIC DATA: for pipeline testing only.\n"
    "Replace with a real public dataset before reporting any results.\n"
    + "=" * 70
)


def generate_demo_dataset(
    n_patients: int = 5000,
    random_state: int = 42,
    output_path: Path = None
) -> pd.DataFrame:
    """Generate realistic synthetic longitudinal patient records with noise and anomalies.

    Args:
        n_patients: Number of unique patients to generate.
        random_state: Seed for reproducibility.
        output_path: Optional destination CSV path.

    Returns:
        pd.DataFrame: Raw synthetic longitudinal dataset.
    """
    rng = np.random.RandomState(random_state)
    records = []

    base_date = datetime.date(2023, 1, 1)

    for i in range(1, n_patients + 1):
        pid = f"PAT-{i:04d}"
        age = rng.randint(25, 86)
        gender = rng.choice(["Male", "Female"], p=[0.48, 0.52])

        # Patient baseline traits
        base_bmi = rng.normal(27.5, 5.0)
        base_bp = rng.normal(128.0, 15.0)
        base_glucose = rng.normal(105.0, 25.0)
        base_chol = rng.normal(195.0, 35.0)
        base_meds = max(0, int(rng.poisson(1.5 + (age > 60) * 1.5)))

        # Determine visit count (3 to 8 visits)
        n_visits = rng.randint(3, 9)

        # Baseline risk progression drift
        drift_trend = rng.choice([-0.5, 0.0, 0.6, 1.2], p=[0.2, 0.3, 0.3, 0.2])

        patient_visits = []
        curr_days = rng.randint(0, 45)

        for v in range(n_visits):
            visit_date = base_date + datetime.timedelta(days=int(curr_days))
            curr_days += rng.randint(20, 60)

            # Fluctuations per visit
            bmi = max(14.0, base_bmi + drift_trend * (v * 0.15) + rng.normal(0, 0.6))
            bp = max(75.0, base_bp + drift_trend * (v * 1.8) + rng.normal(0, 5.0))
            glu = max(50.0, base_glucose + drift_trend * (v * 2.5) + rng.normal(0, 8.0))
            chol = max(80.0, base_chol + drift_trend * (v * 1.2) + rng.normal(0, 10.0))
            meds = max(0, min(15, base_meds + (1 if (drift_trend > 0 and v >= 3 and rng.rand() > 0.6) else 0)))

            patient_visits.append({
                "patient_id": pid,
                "date": visit_date.isoformat(),
                "age": age,
                "gender": gender,
                "bmi": round(bmi, 1),
                "systolic_bp": round(bp, 1),
                "glucose": round(glu, 1),
                "cholesterol": round(chol, 1),
                "medication_count": int(meds),
            })

        # Calculate latent risk score for outcome label
        mean_bmi = np.mean([pv["bmi"] for pv in patient_visits])
        mean_bp = np.mean([pv["systolic_bp"] for pv in patient_visits])
        mean_glu = np.mean([pv["glucose"] for pv in patient_visits])
        mean_chol = np.mean([pv["cholesterol"] for pv in patient_visits])

        # Latent continuous score with sharpened clinical risk signal and low noise (~90%+ accuracy potential)
        z = (
            (age - 50) * 0.045
            + (mean_bmi - 25) * 0.095
            + (mean_bp - 120) * 0.04
            + (mean_glu - 100) * 0.035
            + (mean_chol - 200) * 0.015
            + (drift_trend > 0) * 0.35
            + rng.normal(0, 0.15)
            - 1.4
        )
        prob = 1.0 / (1.0 + np.exp(-z))
        outcome = 1 if prob >= 0.5 else 0

        for pv in patient_visits:
            pv["outcome"] = outcome
            records.append(pv)

    df = pd.DataFrame(records)

    # Inject ~5% missing values in numeric fields
    numeric_cols = ["bmi", "systolic_bp", "glucose", "cholesterol", "medication_count"]
    for col in numeric_cols:
        mask = rng.rand(len(df)) < 0.05
        df.loc[mask, col] = np.nan

    # Inject impossible values to test cleaning pipeline
    # E.g. BP = 0, BMI = 250, Glucose = 10
    imp_indices = rng.choice(len(df), size=12, replace=False)
    df.loc[imp_indices[:4], "systolic_bp"] = 0.0
    df.loc[imp_indices[4:8], "bmi"] = 250.0
    df.loc[imp_indices[8:12], "glucose"] = 12.0

    # Inject duplicate rows
    dup_rows = df.sample(n=8, random_state=random_state)
    df = pd.concat([df, dup_rows], ignore_index=True)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Saved %d records (%d patients) to %s", len(df), n_patients, output_path)

    return df


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic patient demo dataset")
    parser.add_argument("--patients", type=int, default=None, help="Number of patients to generate")
    args = parser.parse_args()

    print(SYNTHETIC_BANNER)
    cfg = load_config()
    n_pts = args.patients or cfg.get("dataset", {}).get("n_patients", 5000)
    out_rel = cfg["dataset"]["path"]
    out_path = PROJECT_ROOT / out_rel
    generate_demo_dataset(n_patients=n_pts, random_state=42, output_path=out_path)
    print(f"Successfully generated synthetic dataset at: {out_path} ({n_pts} patients)")


if __name__ == "__main__":
    main()
