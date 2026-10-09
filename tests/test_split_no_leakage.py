"""Unit tests verifying zero patient leakage across train and test partitions."""

from sklearn.model_selection import GroupShuffleSplit
from src.config import load_config
from src.data_loader import load_raw
from src.feature_engineering import build_patient_features
from src.preprocessing import clean_data


def test_zero_patient_leakage_longitudinal():
    """Verify that in longitudinal mode, no patient appears in both train and test splits."""
    cfg = load_config()
    raw_df = load_raw(cfg)
    cleaned_df, _ = clean_data(raw_df, cfg)
    features_df = build_patient_features(cleaned_df, cfg)

    pid_col = cfg["dataset"]["patient_id_col"]
    target_col = cfg["dataset"]["target_col"]
    groups = features_df[pid_col]
    X = features_df.drop(columns=[target_col])
    y = features_df[target_col].astype(int)

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["random_state"],
    )
    train_idx, test_idx = next(gss.split(X, y, groups))

    train_pids = set(features_df.iloc[train_idx][pid_col])
    test_pids = set(features_df.iloc[test_idx][pid_col])

    overlap = train_pids.intersection(test_pids)
    assert len(overlap) == 0, f"Detected leaking patients: {overlap}"

    # Verify class balance is reasonably preserved (within 15% tolerance)
    train_pos_rate = y.iloc[train_idx].mean()
    test_pos_rate = y.iloc[test_idx].mean()
    assert abs(train_pos_rate - test_pos_rate) < 0.15
