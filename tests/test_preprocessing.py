"""Unit tests for preprocessing and data cleaning rules."""

import numpy as np
import pandas as pd
from src.config import load_config
from src.preprocessing import build_preprocessor, clean_data


def test_transposed_bp_corrected():
    """Verify that inverted blood pressures (e.g. ap_hi=80, ap_lo=120) are automatically swapped."""
    cfg = load_config()
    df = pd.DataFrame([{
        "age": 18000,
        "gender": 1,
        "height": 165,
        "weight": 70.0,
        "ap_hi": 80,
        "ap_lo": 120,
        "cholesterol": 1,
        "gluc": 1,
        "smoke": 0,
        "alco": 0,
        "active": 1,
    }])

    cleaned, report = clean_data(df, cfg)
    assert cleaned.loc[0, "ap_hi"] == 120
    assert cleaned.loc[0, "ap_lo"] == 80
    assert report["bp_transpositions_fixed"] == 1


def test_impossible_values_become_nan():
    """Verify values outside valid_ranges become NaN rather than remaining in dataset."""
    cfg = load_config()
    df = pd.DataFrame([
        {
            "age": 18000,
            "gender": 1,
            "height": 55,         # < 120 -> impossible
            "weight": 70.0,
            "ap_hi": 350,        # > 240 -> impossible
            "ap_lo": 20,         # < 40 -> impossible
            "cholesterol": 1,
            "gluc": 1,
            "smoke": 0,
            "alco": 0,
            "active": 1,
        }
    ])

    cleaned, report = clean_data(df, cfg)
    assert np.isnan(cleaned.loc[0, "height"])
    assert np.isnan(cleaned.loc[0, "ap_hi"])
    assert np.isnan(cleaned.loc[0, "ap_lo"])


def test_duplicates_removed():
    """Verify exact duplicate rows are stripped."""
    cfg = load_config()
    row = {
        "age": 18000,
        "gender": 1,
        "height": 165,
        "weight": 65.0,
        "ap_hi": 120,
        "ap_lo": 80,
        "cholesterol": 1,
        "gluc": 1,
        "smoke": 0,
        "alco": 0,
        "active": 1,
    }
    df = pd.DataFrame([row, row, row])
    cleaned, report = clean_data(df, cfg)
    assert len(cleaned) == 1
    assert report["duplicates_removed"] == 2


def test_pipeline_fit_no_data_leakage():
    """Verify fitted scaler statistics reflect training data only, never test data."""
    cfg = load_config()
    numeric_cols = ["weight"]
    cat_cols = []

    train_df = pd.DataFrame({"weight": [60.0, 80.0]})  # mean = 70.0
    test_df = pd.DataFrame({"weight": [120.0, 160.0]})

    prep = build_preprocessor(cfg, numeric_cols, cat_cols)
    prep.fit(train_df)

    fitted_mean = prep.named_transformers_["num"].named_steps["scaler"].mean_[0]
    assert fitted_mean == 70.0
    assert fitted_mean != float(np.mean([60.0, 80.0, 120.0, 160.0]))
