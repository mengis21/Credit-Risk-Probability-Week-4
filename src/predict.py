"""Inference helpers for the credit risk service."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import mlflow
import mlflow.sklearn  # noqa: F401 - ensure sklearn flavor registered
import pandas as pd

from .data_processing import CUSTOMER_ID_COL, build_inference_features


DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:mlruns")
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "credit-risk-probability-model")
DEFAULT_MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")


@lru_cache(maxsize=1)
def _load_model(model_name: str, model_stage: str) -> object:
    mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)
    model_uri = f"models:/{model_name}/{model_stage}"
    return mlflow.sklearn.load_model(model_uri)


def predict_probabilities(
    transactions: pd.DataFrame,
    snapshot_date: Optional[pd.Timestamp] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    model_stage: str = DEFAULT_MODEL_STAGE,
) -> pd.DataFrame:
    """Score customers and return probability estimates."""

    if transactions.empty:
        raise ValueError("No transactions supplied for scoring.")

    features = build_inference_features(transactions, snapshot_date=snapshot_date)
    model = _load_model(model_name, model_stage)

    feature_matrix = features.drop(columns=[CUSTOMER_ID_COL])
    probabilities = model.predict_proba(feature_matrix)[:, 1]

    return pd.DataFrame(
        {
            CUSTOMER_ID_COL: features[CUSTOMER_ID_COL].values,
            "probability": probabilities,
        }
    )
