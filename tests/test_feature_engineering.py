"""Unit tests for feature engineering and longitudinal aggregation."""

import numpy as np
import pandas as pd
import pytest
from src.config import load_config
from src.feature_engineering import build_patient_features


def test_bp_worked_example():
    """Verify worked BP example from Section 6.4:

    Visits: [125, 130, 138, 145] ->
    latest=145, first=125, mean=134.5, change=20, pct_change=0.16, std≈8.81
    """
    cfg = load_config()
    df = pd.DataFrame({
        "patient_id": ["P1", "P1", "P1", "P1"],
        "date": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"],
        "age": [50, 50, 50, 50],
        "gender": ["Female", "Female", "Female", "Female"],
        "bmi": [25.0, 25.0, 25.0, 25.0],
        "systolic_bp": [125.0, 130.0, 138.0, 145.0],
        "glucose": [100.0, 100.0, 100.0, 100.0],
        "cholesterol": [200.0, 200.0, 200.0, 200.0],
        "medication_count": [1, 1, 1, 1],
        "outcome": [0, 0, 0, 1],
    })

    feats = build_patient_features(df, cfg)
    assert len(feats) == 1

    row = feats.iloc[0]
    assert row["systolic_bp_latest"] == 145.0
    assert row["systolic_bp_first"] == 125.0
    assert row["systolic_bp_mean"] == 134.5
    assert row["systolic_bp_change"] == 20.0
    assert pytest.approx(row["systolic_bp_pct_change"], 0.001) == 0.16
    assert pytest.approx(row["systolic_bp_std"], 0.05) == 8.81


def test_single_record_patient():
    """A patient with a single observation should yield std=0, change=0, pct_change=0."""
    cfg = load_config()
    df = pd.DataFrame({
        "patient_id": ["P_SINGLE"],
        "date": ["2024-01-01"],
        "age": [62],
        "gender": ["Male"],
        "bmi": [28.0],
        "systolic_bp": [135.0],
        "glucose": [110.0],
        "cholesterol": [215.0],
        "medication_count": [2],
        "outcome": [1],
    })

    feats = build_patient_features(df, cfg)
    assert len(feats) == 1
    row = feats.iloc[0]
    assert row["systolic_bp_latest"] == 135.0
    assert row["systolic_bp_first"] == 135.0
    assert row["systolic_bp_change"] == 0.0
    assert row["systolic_bp_pct_change"] == 0.0
    assert row["systolic_bp_std"] == 0.0
    assert row["span_days"] == 0.0
    assert row["n_records"] == 1


def test_one_row_per_patient():
    """Verify grouping guarantees strictly one row per patient."""
    cfg = load_config()
    df = pd.DataFrame({
        "patient_id": ["P1", "P1", "P2", "P2", "P3"],
        "date": ["2024-01-01", "2024-02-01", "2024-01-10", "2024-03-01", "2024-02-15"],
        "age": [40, 40, 50, 50, 60],
        "gender": ["Male", "Male", "Female", "Female", "Male"],
        "bmi": [24.0, 24.5, 29.0, 29.2, 31.0],
        "systolic_bp": [120.0, 122.0, 140.0, 144.0, 130.0],
        "glucose": [90.0, 92.0, 120.0, 125.0, 115.0],
        "cholesterol": [180.0, 185.0, 220.0, 225.0, 205.0],
        "medication_count": [0, 0, 2, 2, 1],
        "outcome": [0, 0, 1, 1, 0],
    })

    feats = build_patient_features(df, cfg)
    assert len(feats) == 3
    assert set(feats["patient_id"]) == {"P1", "P2", "P3"}


def test_missing_age_across_visits():
    """Verify that a patient with some missing age records still resolves valid age from other visits."""
    cfg = load_config()
    df = pd.DataFrame({
        "patient_id": ["P1", "P1", "P1"],
        "date": ["2024-01-01", "2024-02-01", "2024-03-01"],
        "age": [np.nan, 55.0, np.nan],
        "gender": ["F", "F", "F"],
        "bmi": [26.0, 26.5, 27.0],
        "systolic_bp": [130.0, 132.0, 134.0],
        "glucose": [105.0, 110.0, 115.0],
        "cholesterol": [210.0, 215.0, 220.0],
        "medication_count": [1, 1, 1],
        "outcome": [1, 1, 1],
    })

    feats = build_patient_features(df, cfg)
    assert len(feats) == 1
    row = feats.iloc[0]
    assert row["age"] == 55.0
    assert not np.isnan(row["age_x_glucose"])
    assert not np.isnan(row["poly_age2"])
    assert row["poly_age2"] == 55.0 ** 2
