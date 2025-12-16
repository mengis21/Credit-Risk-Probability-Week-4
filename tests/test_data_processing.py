"""Unit tests for feature engineering and preprocessing."""
from __future__ import annotations

import pandas as pd

from src.data_processing import (
    CUSTOMER_ID_COL,
    build_preprocessing_pipeline,
    build_training_dataset,
)


def _sample_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionId": ["t1", "t2", "t3", "t4", "t5", "t6"],
            "CustomerId": ["c1", "c1", "c2", "c2", "c3", "c3"],
            "Amount": [1000, 2000, 500, 700, 300, 450],
            "Value": [1000, 2000, 500, 700, 300, 450],
            "TransactionStartTime": pd.to_datetime(
                [
                    "2025-01-01T10:00:00Z",
                    "2025-01-03T11:00:00Z",
                    "2025-01-02T09:30:00Z",
                    "2025-01-04T08:00:00Z",
                    "2025-01-05T14:45:00Z",
                    "2025-01-06T16:20:00Z",
                ]
            ),
            "ProductCategory": [
                "airtime",
                "financial_services",
                "airtime",
                "utility",
                "financial_services",
                "airtime",
            ],
            "ChannelId": ["web", "mobile", "web", "mobile", "mobile", "web"],
            "ProviderId": ["p1", "p2", "p1", "p3", "p4", "p4"],
            "PricingStrategy": [1, 1, 2, 2, 3, 1],
        }
    )


def test_build_training_dataset_contains_expected_columns() -> None:
    df = _sample_transactions()

    features, target, metadata = build_training_dataset(df)

    assert CUSTOMER_ID_COL in features.columns
    assert {"rfm_recency", "rfm_frequency", "rfm_monetary"}.issubset(features.columns)
    assert target.isin({0, 1}).all()
    assert metadata["rfm_cluster"].nunique() <= 3


def test_preprocessing_pipeline_transforms_feature_matrix() -> None:
    df = _sample_transactions()
    features, target, _ = build_training_dataset(df)

    pipeline = build_preprocessing_pipeline()
    transformed = pipeline.fit_transform(features.drop(columns=[CUSTOMER_ID_COL]), target)

    assert transformed.shape[0] == len(features)
    assert transformed.ndim == 2
