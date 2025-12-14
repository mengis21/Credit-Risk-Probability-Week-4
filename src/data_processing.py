"""Data preprocessing utilities for the credit risk model.

This module will be expanded during Tasks 3-5. For now it exposes a
placeholder pipeline factory used by the unit test scaffold.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


def build_baseline_dataframe(df: pd.DataFrame, limit: Optional[int] = None) -> pd.DataFrame:
    """Return a trimmed copy of the incoming dataframe for exploratory work.

    Parameters
    ----------
    df: pd.DataFrame
        Raw transaction records.
    limit: Optional[int]
        Optional row limit for lightweight experimentation.
    """

    if limit is not None:
        return df.head(limit).copy()
    return df.copy()
