"""Unit tests verifying stratified holdout splitting integrity and class balance."""

import numpy as np
from sklearn.model_selection import train_test_split
from src.config import load_config
from src.data_loader import load_raw
from src.feature_engineering import build_patient_features
from src.preprocessing import clean_data


def test_stratified_split_integrity():
    """Verify that train/test partition maintains class balance and zero index leakage."""
    cfg = load_config()
    raw_df = load_raw(cfg)
    cleaned_df, _ = clean_data(raw_df, cfg)
    features_df = build_patient_features(cleaned_df, cfg)

    target_col = cfg["dataset"]["target_col"]
    y = features_df[target_col].astype(int)
    X = features_df.drop(columns=[target_col])

    indices = np.arange(len(features_df))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["random_state"],
        stratify=y if cfg["split"].get("stratify") else None,
    )

    # Zero index overlap
    overlap = set(train_idx).intersection(set(test_idx))
    assert len(overlap) == 0, f"Detected leaking indices: {overlap}"

    # Verify class balance is preserved (within 2% tolerance)
    train_pos_rate = y.iloc[train_idx].mean()
    test_pos_rate = y.iloc[test_idx].mean()
    assert abs(train_pos_rate - test_pos_rate) < 0.02
