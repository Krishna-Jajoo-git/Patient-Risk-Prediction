"""Inference module for loading trained artifacts and executing predictions."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import PROJECT_ROOT, load_config
from src.data_loader import validate_schema
from src.feature_engineering import build_patient_features, get_feature_lists, summarize_patient
from src.preprocessing import clean_data


def load_artifacts() -> Tuple[Pipeline, Dict[str, Any], Dict[str, Any]]:
    """Load the trained model pipeline, metadata, and performance metrics.

    Returns:
        tuple[Pipeline, dict, dict]: (best_pipeline, model_metadata, metrics)

    Raises:
        FileNotFoundError: If the model artifacts have not yet been produced.
    """
    cfg = load_config()
    models_dir = PROJECT_ROOT / cfg["paths"]["models_dir"]
    model_path = models_dir / "best_model.joblib"
    meta_path = models_dir / "model_metadata.json"
    metrics_path = models_dir / "metrics.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "Please run 'python -m src.train' to train the pipeline first."
        )

    model = joblib.load(model_path)

    metadata = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    return model, metadata, metrics


def validate_input(df: pd.DataFrame, cfg: Dict[str, Any]) -> List[str]:
    """Validate uploaded patient data schema without requiring target label."""
    return validate_schema(df, cfg, require_target=False)


def predict_from_history(
    history_df: pd.DataFrame, cfg: Dict[str, Any], model: Pipeline
) -> Dict[str, Any]:
    """Process a single patient's raw medical history records and compute prediction.

    Pipeline:
      1. Clean input data (impossible ranges, format standardization)
      2. Aggregate longitudinal trajectory into single patient feature vector
      3. Compute prediction & calibrated probabilities via the saved Pipeline
      4. Extract clinical summary cards

    Returns:
        dict: Prediction results, label text, probabilities, feature vector, and vital summaries.
    """
    if history_df.empty:
        raise ValueError("Cannot predict on empty patient history.")

    cleaned_df, _ = clean_data(history_df, cfg)
    features_df = build_patient_features(cleaned_df, cfg)

    num_cols, cat_cols = get_feature_lists(cfg)
    expected_features = num_cols + cat_cols

    # Ensure all expected columns are present (filling missing with NaN for imputer to handle)
    for col in expected_features:
        if col not in features_df.columns:
            features_df[col] = np.nan

    X = features_df[expected_features]

    pred = int(model.predict(X)[0])

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X)[0]
        # In binary classification, probability of positive/elevated risk
        prob = float(probas[1]) if len(probas) > 1 else float(probas[0])
        class_probas = [float(p) for p in probas]
    else:
        prob = 1.0 if pred == 1 else 0.0
        class_probas = [1.0 - prob, prob]

    task = cfg["dataset"].get("task", "binary")
    if task == "binary":
        label_text = "Elevated Health Risk" if pred == 1 else "Normal / Low Risk"
        risk_level = "High" if prob >= 0.65 else ("Moderate" if prob >= 0.40 else "Low")
    else:
        label_text = f"Class {pred}"
        risk_level = "Categorical"

    summary = summarize_patient(cleaned_df, cfg)
    features_dict = X.iloc[0].to_dict()

    return {
        "prediction": pred,
        "label_text": label_text,
        "risk_level": risk_level,
        "probability": round(prob, 4),
        "class_probabilities": class_probas,
        "features_used": features_dict,
        "patient_summary": summary,
    }


def predict_from_features(features_df: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    """Batch prediction on pre-engineered feature table."""
    cfg = load_config()
    num_cols, cat_cols = get_feature_lists(cfg)
    expected = num_cols + cat_cols

    for c in expected:
        if c not in features_df.columns:
            features_df[c] = np.nan

    X = features_df[expected]
    preds = model.predict(X)
    out = features_df.copy()
    out["prediction"] = preds

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X)
        if probas.shape[1] == 2:
            out["risk_probability"] = probas[:, 1]

    return out
