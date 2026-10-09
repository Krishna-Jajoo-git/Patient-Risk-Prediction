"""Unit tests for prediction roundtrip and input validation."""

import pandas as pd
import pytest
from src.config import load_config
from src.predict import load_artifacts, predict_from_history, validate_input


def test_predict_from_history_roundtrip():
    """Verify that predict_from_history returns valid probabilities and risk labels."""
    cfg = load_config()
    model, _, _ = load_artifacts()

    sample_history = pd.DataFrame({
        "patient_id": ["TEST-01", "TEST-01", "TEST-01"],
        "date": ["2024-01-01", "2024-02-15", "2024-03-30"],
        "age": [58, 58, 58],
        "gender": ["Male", "Male", "Male"],
        "bmi": [29.5, 30.1, 30.8],
        "systolic_bp": [138.0, 142.0, 146.0],
        "glucose": [115.0, 122.0, 128.0],
        "cholesterol": [210.0, 215.0, 225.0],
        "medication_count": [2, 2, 3],
    })

    result = predict_from_history(sample_history, cfg, model)

    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["label_text"] in ("Elevated Health Risk", "Normal / Low Risk")
    assert result["risk_level"] in ("High", "Moderate", "Low")
    assert "systolic_bp" in result["patient_summary"]["vitals"]


def test_invalid_schema_returns_errors_not_exceptions():
    """Verify that corrupted input schema cleanly produces descriptive error list."""
    cfg = load_config()
    bad_df = pd.DataFrame({"unknown_col": [1, 2, 3]})
    errors = validate_input(bad_df, cfg)
    assert len(errors) > 0
    assert any("Missing required column" in err for err in errors)


def test_predict_from_history_with_missing_and_abbreviated_gender():
    """Verify inference pipeline operates robustly with partially missing age and 'F'/'Other' gender."""
    import numpy as np
    cfg = load_config()
    model, _, _ = load_artifacts()

    sample_history = pd.DataFrame({
        "patient_id": ["P-SAMPLE", "P-SAMPLE"],
        "date": ["2021-05-01", "2021-08-15"],
        "age": [np.nan, 62.0],
        "gender": ["F", "F"],
        "bmi": [27.4, 28.1],
        "systolic_bp": [135.0, 140.0],
        "glucose": [120.0, np.nan],
        "cholesterol": [205.0, 210.0],
        "medication_count": [1, 2],
    })

    result = predict_from_history(sample_history, cfg, model)
    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["risk_level"] in ("High", "Moderate", "Low")
    assert result["features_used"]["age"] == 62.0
