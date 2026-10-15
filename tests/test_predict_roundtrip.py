"""Unit tests for prediction roundtrip, batch scoring, and input validation."""

import pandas as pd
import pytest
from src.config import load_config
from src.predict import load_artifacts, predict_from_features, predict_patient, validate_input


def test_predict_patient_roundtrip():
    """Verify that predict_patient returns valid probabilities, classifications, and clinical summaries."""
    cfg = load_config()
    model, _, _ = load_artifacts()

    sample_patient = pd.DataFrame([{
        "age": 55,
        "gender": 2,
        "height": 175,
        "weight": 86.0,
        "ap_hi": 140,
        "ap_lo": 90,
        "cholesterol": 2,
        "gluc": 1,
        "smoke": 1,
        "alco": 0,
        "active": 0,
    }])

    result = predict_patient(sample_patient, cfg, model)

    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["risk_level"] in ("High", "Moderate", "Low")
    assert "demographics" in result["patient_summary"]
    assert "hemodynamics" in result["patient_summary"]
    assert result["patient_summary"]["hemodynamics"]["ap_hi"] == 140.0
    assert result["patient_summary"]["hemodynamics"]["bp_category"] == "Stage 2 HTN"


def test_invalid_schema_returns_errors():
    """Verify that corrupted input schema cleanly produces descriptive error list."""
    cfg = load_config()
    bad_df = pd.DataFrame({"unknown_col": [1, 2, 3]})
    errors = validate_input(bad_df, cfg)
    assert len(errors) > 0
    assert any("Missing required column" in err for err in errors)


def test_predict_batch_features():
    """Verify batch scoring appends predictions and probabilities to all rows."""
    model, _, _ = load_artifacts()
    batch_df = pd.DataFrame([
        {"age": 45, "gender": 1, "height": 160, "weight": 55, "ap_hi": 110, "ap_lo": 70, "cholesterol": 1, "gluc": 1, "smoke": 0, "alco": 0, "active": 1},
        {"age": 60, "gender": 2, "height": 178, "weight": 95, "ap_hi": 155, "ap_lo": 95, "cholesterol": 3, "gluc": 2, "smoke": 1, "alco": 1, "active": 0},
    ])

    scored = predict_from_features(batch_df, model)
    assert len(scored) == 2
    assert "prediction" in scored.columns
    assert "risk_probability" in scored.columns
    assert "risk_level" in scored.columns
