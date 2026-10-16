"""Unit tests for data_loader and run_eda utilities."""

import pandas as pd
import pytest
from src.data_loader import run_eda


def test_run_eda_contains_all_requested_fields():
    """Verify that run_eda computes and returns all 8 requested EDA components."""
    data = {
        "patient_id": ["P1", "P1", "P2", "P2", "P2"],
        "age": [45, 45, 60, 60, None],
        "systolic_bp": [120.0, 120.0, 140.0, 140.0, 130.0],
        "outcome": [0, 0, 1, 1, 1],
    }
    df = pd.DataFrame(data)

    results = run_eda(df, verbose=True)

    # 1. df.shape
    assert results["shape"] == (5, 4)

    # 2. df.columns
    assert list(results["columns"]) == ["patient_id", "age", "systolic_bp", "outcome"]

    # 3. df.head()
    assert len(results["head"]) == 5

    # 4. df.info()
    assert "patient_id" in results["info"]
    assert "systolic_bp" in results["info"]

    # 5. df.describe()
    assert "age" in results["describe"].columns
    assert "systolic_bp" in results["describe"].columns

    # 6. df.isnull().sum()
    assert results["isnull_sum"]["age"] == 1
    assert results["isnull_sum"]["patient_id"] == 0

    # 7. df.duplicated().sum()
    assert results["duplicated_sum"] == 2

    # 8. df.drop_duplicates()
    assert len(results["df_dedup"]) == 3
    assert results["df_dedup"].duplicated().sum() == 0
