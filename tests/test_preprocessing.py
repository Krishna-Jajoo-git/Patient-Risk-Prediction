"""Unit tests for preprocessing and data cleaning rules."""

import numpy as np
import pandas as pd
from src.config import load_config
from src.preprocessing import build_preprocessor, clean_data


def test_impossible_values_become_nan():
    """Verify values outside valid_ranges become NaN rather than remaining in dataset."""
    cfg = load_config()
    df = pd.DataFrame({
        "patient_id": ["P1", "P2"],
        "date": ["2024-01-01", "2024-01-02"],
        "age": [50, 45],
        "gender": ["Male", "Female"],
        "bmi": [25.0, 250.0],       # 250 is outside [10, 70]
        "systolic_bp": [0.0, 130.0],  # 0 is outside [60, 260]
        "glucose": [100.0, 100.0],
        "cholesterol": [200.0, 200.0],
        "medication_count": [1, 1],
    })

    cleaned, report = clean_data(df, cfg)
    assert np.isnan(cleaned.loc[0, "systolic_bp"])
    assert cleaned.loc[1, "systolic_bp"] == 130.0
    assert np.isnan(cleaned.loc[1, "bmi"])
    assert report["impossible_values_nulled"]["systolic_bp"] == 1
    assert report["impossible_values_nulled"]["bmi"] == 1


def test_duplicates_removed():
    """Verify exact duplicate rows are stripped."""
    cfg = load_config()
    row = {
        "patient_id": "P1",
        "date": "2024-01-01",
        "age": 50,
        "gender": "Male",
        "bmi": 25.0,
        "systolic_bp": 120.0,
        "glucose": 95.0,
        "cholesterol": 180.0,
        "medication_count": 0,
    }
    df = pd.DataFrame([row, row, row])
    cleaned, report = clean_data(df, cfg)
    assert len(cleaned) == 1
    assert report["duplicates_removed"] == 2


def test_preprocessor_handles_unseen_categories():
    """Ensure OneHotEncoder in preprocessor handles unknown categorical levels gracefully."""
    cfg = load_config()
    numeric_cols = ["age", "bmi"]
    cat_cols = ["gender"]

    train_df = pd.DataFrame({"age": [30, 40], "bmi": [22.0, 26.0], "gender": ["Male", "Female"]})
    test_df = pd.DataFrame({"age": [50], "bmi": [28.0], "gender": ["UnknownGender"]})

    prep = build_preprocessor(cfg, numeric_cols, cat_cols)
    prep.fit(train_df)
    transformed_test = prep.transform(test_df)
    assert transformed_test.shape[0] == 1
    assert not np.isnan(transformed_test).any()


def test_pipeline_fit_no_data_leakage():
    """Verify fitted scaler statistics reflect training data only, never test data."""
    cfg = load_config()
    numeric_cols = ["bmi"]
    cat_cols = []

    train_df = pd.DataFrame({"bmi": [20.0, 30.0]})  # mean = 25.0
    test_df = pd.DataFrame({"bmi": [100.0, 200.0]})

    prep = build_preprocessor(cfg, numeric_cols, cat_cols)
    prep.fit(train_df)

    fitted_mean = prep.named_transformers_["num"].named_steps["scaler"].mean_[0]
    assert fitted_mean == 25.0
    # Scaler mean must NOT be contaminated by test values
    assert fitted_mean != float(np.mean([20.0, 30.0, 100.0, 200.0]))


def test_gender_standardization():
    """Verify that gender representations (Male, Female, M, F, Other) normalize consistently."""
    cfg = load_config()
    df = pd.DataFrame({
        "patient_id": ["P1", "P2", "P3", "P4", "P5"],
        "date": ["2024-01-01"] * 5,
        "age": [30, 40, 50, 60, 70],
        "gender": ["male", "Female", "M", "f", "other"],
        "bmi": [22.0, 24.0, 26.0, 28.0, 30.0],
        "systolic_bp": [120.0, 122.0, 124.0, 126.0, 128.0],
        "glucose": [90.0, 95.0, 100.0, 105.0, 110.0],
        "cholesterol": [180.0, 185.0, 190.0, 195.0, 200.0],
        "medication_count": [0, 1, 1, 2, 2],
    })

    cleaned, _ = clean_data(df, cfg)
    assert cleaned["gender"].tolist() == ["M", "F", "M", "F", "Other"]
