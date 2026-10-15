"""Unit tests for cardiovascular domain feature engineering."""

import numpy as np
import pandas as pd
import pytest
from src.config import load_config
from src.feature_engineering import build_patient_features


def test_bmi_and_pulse_pressure_derivation():
    """Verify worked BMI and Pulse Pressure calculation:
    Height=165 cm, Weight=68 kg -> BMI=24.98 kg/m^2 (Normal)
    SBP=135 mmHg, DBP=85 mmHg -> Pulse Pressure=50 mmHg, MAP=101.67 mmHg (Stage 1 HTN)
    """
    cfg = load_config()
    df = pd.DataFrame([{
        "age": 18262,  # ~50 years
        "gender": 1,
        "height": 165,
        "weight": 68.0,
        "ap_hi": 135,
        "ap_lo": 85,
        "cholesterol": 1,
        "gluc": 1,
        "smoke": 0,
        "alco": 0,
        "active": 1,
        "cardio": 0,
    }])

    feats = build_patient_features(df, cfg)
    assert len(feats) == 1
    row = feats.iloc[0]

    assert pytest.approx(row["age_years"], 0.1) == 50.0
    assert pytest.approx(row["bmi"], 0.05) == 24.98
    assert row["pulse_pressure"] == 50.0
    assert pytest.approx(row["map"], 0.1) == 101.7
    assert row["bp_category"] == "Stage 1 HTN"
    assert row["bmi_category"] == "Normal"


def test_bp_stage_classification():
    """Verify AHA blood pressure stage classifications across normal, elevated, and hypertensive patients."""
    cfg = load_config()
    df = pd.DataFrame([
        {"age": 45, "gender": 1, "height": 160, "weight": 60, "ap_hi": 115, "ap_lo": 75, "cholesterol": 1, "gluc": 1, "smoke": 0, "alco": 0, "active": 1},
        {"age": 45, "gender": 1, "height": 160, "weight": 60, "ap_hi": 125, "ap_lo": 78, "cholesterol": 1, "gluc": 1, "smoke": 0, "alco": 0, "active": 1},
        {"age": 45, "gender": 1, "height": 160, "weight": 60, "ap_hi": 145, "ap_lo": 92, "cholesterol": 1, "gluc": 1, "smoke": 0, "alco": 0, "active": 1},
        {"age": 45, "gender": 1, "height": 160, "weight": 60, "ap_hi": 185, "ap_lo": 125, "cholesterol": 1, "gluc": 1, "smoke": 0, "alco": 0, "active": 1},
    ])

    feats = build_patient_features(df, cfg)
    stages = feats["bp_category"].tolist()
    assert stages == ["Normal", "Elevated", "Stage 2 HTN", "Crisis"]


def test_age_years_conversion():
    """Verify that age in days (>120) is converted to years, while age in years is kept as-is."""
    cfg = load_config()
    df = pd.DataFrame([
        {"age": 20000, "gender": 2, "height": 175, "weight": 80, "ap_hi": 120, "ap_lo": 80, "cholesterol": 1, "gluc": 1, "smoke": 0, "alco": 0, "active": 1},
        {"age": 55, "gender": 2, "height": 175, "weight": 80, "ap_hi": 120, "ap_lo": 80, "cholesterol": 1, "gluc": 1, "smoke": 0, "alco": 0, "active": 1},
    ])

    feats = build_patient_features(df, cfg)
    assert pytest.approx(feats.iloc[0]["age_years"], 0.1) == 54.8
    assert feats.iloc[1]["age_years"] == 55.0
