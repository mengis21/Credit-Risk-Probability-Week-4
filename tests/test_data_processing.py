"""Unit test placeholders for data processing."""
import pandas as pd

from src.data_processing import build_baseline_dataframe


def test_build_baseline_dataframe_returns_copy():
    df = pd.DataFrame({"value": [1, 2, 3]})

    result = build_baseline_dataframe(df)

    assert not result is df
    assert result.equals(df)


def test_build_baseline_dataframe_applies_limit():
    df = pd.DataFrame({"value": [1, 2, 3]})

    result = build_baseline_dataframe(df, limit=2)

    assert len(result) == 2
