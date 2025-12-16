"""Feature engineering helpers for the credit risk probability project.

This module centralises transaction feature engineering, proxy target
construction, and preprocessing pipeline creation. It is intentionally
framework-agnostic so it can serve both offline training and the FastAPI
inference service.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

CUSTOMER_ID_COL = "CustomerId"
DATETIME_COL = "TransactionStartTime"
AMOUNT_COL = "Amount"
VALUE_COL = "Value"

NUMERIC_FEATURES: Sequence[str] = (
    "total_transaction_amount",
    "average_transaction_amount",
    "transaction_count",
    "transaction_amount_std",
    "total_transaction_value",
    "average_transaction_value",
    "transaction_value_std",
    "transaction_hour_mean",
    "transaction_hour_std",
    "transaction_day_mean",
    "transaction_day_std",
    "transaction_month_mean",
    "transaction_month_std",
    "provider_nunique",
    "product_nunique",
    "channel_nunique",
    "pricing_strategy_mean",
    "rfm_recency",
    "rfm_frequency",
    "rfm_monetary",
)

WOE_FEATURES: Sequence[str] = (
    "primary_product_category",
    "primary_channel_id",
    "primary_provider_id",
)

CATEGORICAL_FEATURES: Sequence[str] = ("pricing_strategy_mode",)


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def _ensure_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _mode(series: pd.Series) -> Optional[str]:
    modes = series.dropna().mode()
    return modes.iloc[0] if not modes.empty else None


# -----------------------------------------------------------------------------
# WoE encoder
# -----------------------------------------------------------------------------


@dataclass
class CategoryStats:
    woe: Dict[str, float]
    information_value: float


class WoEEncoder(BaseEstimator, TransformerMixin):
    """Simple Weight-of-Evidence encoder with Information Value tracking."""

    def __init__(self, features: Sequence[str], smoothing: float = 0.5) -> None:
        self.features = tuple(features)
        self.smoothing = smoothing
        self._encodings: Dict[str, CategoryStats] = {}

    @property
    def information_values(self) -> Dict[str, float]:
        return {feature: stats.information_value for feature, stats in self._encodings.items()}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoEEncoder":
        if not isinstance(X, pd.DataFrame):  # pragma: no cover - defensive
            X = pd.DataFrame(X, columns=list(self.features))

        target = pd.Series(y).astype(int)
        for feature in self.features:
            feature_series = X[feature].fillna("__MISSING__").astype(str)
            group = feature_series.to_frame(name=feature).assign(target=target.values)
            agg = (
                group.groupby(feature)["target"]
                .agg(["sum", "count"])
                .rename(columns={"sum": "bad", "count": "total"})
            )
            agg["good"] = agg["total"] - agg["bad"]

            bad_total = agg["bad"].sum()
            good_total = agg["good"].sum()

            bad_rate = (agg["bad"] + self.smoothing) / (bad_total + self.smoothing * len(agg))
            good_rate = (agg["good"] + self.smoothing) / (good_total + self.smoothing * len(agg))

            woe = np.log((bad_rate / good_rate).clip(lower=1e-6))
            iv = ((bad_rate - good_rate) * woe).sum()

            mapping = woe.to_dict()
            self._encodings[feature] = CategoryStats(mapping, float(iv))

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):  # pragma: no cover - defensive
            raise TypeError("WoEEncoder expects a pandas DataFrame as input.")

        transformed = X.copy()
        for feature, stats in self._encodings.items():
            mapping = stats.woe
            transformed[feature] = (
                transformed[feature]
                .fillna("__MISSING__")
                .astype(str)
                .map(mapping)
                .fillna(0.0)
                .astype(float)
            )

        return transformed


# -----------------------------------------------------------------------------
# Feature engineering
# -----------------------------------------------------------------------------


def load_transactions(path: Path | str) -> pd.DataFrame:
    """Load raw transactions from disk with standard parsing."""

    csv_path = Path(path)
    df = pd.read_csv(csv_path, parse_dates=[DATETIME_COL])
    return df


def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched[DATETIME_COL] = _ensure_datetime(enriched[DATETIME_COL])
    enriched["transaction_hour"] = enriched[DATETIME_COL].dt.hour
    enriched["transaction_day"] = enriched[DATETIME_COL].dt.day
    enriched["transaction_month"] = enriched[DATETIME_COL].dt.month
    enriched["transaction_year"] = enriched[DATETIME_COL].dt.year
    return enriched


def engineer_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction-level data into customer-level features."""

    required = {CUSTOMER_ID_COL, AMOUNT_COL, VALUE_COL, DATETIME_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    enriched = add_datetime_features(df)

    optional_defaults = {
        "ProviderId": "provider_missing",
        "ProductCategory": "product_missing",
        "ChannelId": "channel_missing",
        "PricingStrategy": 0,
    }
    for column, default in optional_defaults.items():
        if column not in enriched.columns:
            enriched[column] = default

    grouped = enriched.groupby(CUSTOMER_ID_COL)

    aggregated = grouped.agg(
        total_transaction_amount=(AMOUNT_COL, "sum"),
        average_transaction_amount=(AMOUNT_COL, "mean"),
        transaction_count=(AMOUNT_COL, "count"),
        transaction_amount_std=(AMOUNT_COL, "std"),
        total_transaction_value=(VALUE_COL, "sum"),
        average_transaction_value=(VALUE_COL, "mean"),
        transaction_value_std=(VALUE_COL, "std"),
        transaction_hour_mean=("transaction_hour", "mean"),
        transaction_hour_std=("transaction_hour", "std"),
        transaction_day_mean=("transaction_day", "mean"),
        transaction_day_std=("transaction_day", "std"),
        transaction_month_mean=("transaction_month", "mean"),
        transaction_month_std=("transaction_month", "std"),
        primary_product_category=("ProductCategory", lambda x: _mode(x)),
        primary_channel_id=("ChannelId", lambda x: _mode(x)),
        primary_provider_id=("ProviderId", lambda x: _mode(x)),
        pricing_strategy_mode=("PricingStrategy", lambda x: _mode(x)),
        provider_nunique=("ProviderId", "nunique"),
        product_nunique=("ProductCategory", "nunique"),
        channel_nunique=("ChannelId", "nunique"),
        pricing_strategy_mean=("PricingStrategy", "mean"),
    )

    aggregated = aggregated.reset_index()

    fill_zero_cols = [
        "transaction_amount_std",
        "transaction_value_std",
        "transaction_hour_std",
        "transaction_day_std",
        "transaction_month_std",
    ]
    aggregated[fill_zero_cols] = aggregated[fill_zero_cols].fillna(0.0)
    if "pricing_strategy_mean" in aggregated.columns:
        aggregated["pricing_strategy_mean"] = aggregated["pricing_strategy_mean"].fillna(0.0)
    if "pricing_strategy_mode" in aggregated.columns:
        aggregated["pricing_strategy_mode"] = aggregated["pricing_strategy_mode"].astype(str)

    return aggregated.sort_values(CUSTOMER_ID_COL).reset_index(drop=True)


def compute_rfm(df: pd.DataFrame, snapshot_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Calculate Recency, Frequency, Monetary metrics per customer."""

    enriched = add_datetime_features(df)
    if snapshot_date is None:
        snapshot_date = enriched[DATETIME_COL].max() + pd.Timedelta(days=1)

    grouped = enriched.groupby(CUSTOMER_ID_COL)
    recency = (snapshot_date - grouped[DATETIME_COL].max()).dt.days
    frequency = grouped.size()
    monetary = grouped[VALUE_COL].sum()

    rfm = pd.DataFrame(
        {
            CUSTOMER_ID_COL: recency.index,
            "rfm_recency": recency.values,
            "rfm_frequency": frequency.values,
            "rfm_monetary": monetary.values,
        }
    )
    return rfm.sort_values(CUSTOMER_ID_COL).reset_index(drop=True)


def assign_high_risk_labels(
    rfm: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, KMeans, StandardScaler]:
    """Cluster customers using RFM metrics and derive a proxy label."""

    scaler = StandardScaler()
    features = rfm[["rfm_recency", "rfm_frequency", "rfm_monetary"]]
    scaled = scaler.fit_transform(features)

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = model.fit_predict(scaled)

    enriched = rfm.copy()
    enriched["rfm_cluster"] = clusters

    cluster_stats = (
        enriched.groupby("rfm_cluster")[
            ["rfm_recency", "rfm_frequency", "rfm_monetary"]
        ]
        .mean()
        .assign(
            risk_score=lambda df: df["rfm_recency"].rank(ascending=False)
            + df["rfm_frequency"].rank(ascending=True)
            + df["rfm_monetary"].rank(ascending=True)
        )
    )

    high_risk_cluster = int(cluster_stats["risk_score"].idxmax())
    enriched["is_high_risk"] = (enriched["rfm_cluster"] == high_risk_cluster).astype(int)

    return enriched, model, scaler


def build_training_dataset(
    df: pd.DataFrame,
    snapshot_date: Optional[pd.Timestamp] = None,
    n_clusters: int = 3,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Prepare feature matrix, target vector, and metadata for modelling."""

    features = engineer_customer_features(df)
    rfm = compute_rfm(df, snapshot_date=snapshot_date)
    rfm_with_labels, _, _ = assign_high_risk_labels(rfm, n_clusters=n_clusters)

    dataset = features.merge(rfm_with_labels, on=CUSTOMER_ID_COL, how="inner")
    dataset = dataset.sort_values(CUSTOMER_ID_COL).reset_index(drop=True)

    metadata_cols = [CUSTOMER_ID_COL, "rfm_cluster"]
    metadata = dataset[metadata_cols]

    y = dataset.pop("is_high_risk")
    dataset = dataset.drop(columns=["rfm_cluster"])

    return dataset, y.astype(int), metadata


def build_inference_features(
    df: pd.DataFrame,
    snapshot_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Prepare the feature matrix required by the trained models."""

    features = engineer_customer_features(df)
    rfm = compute_rfm(df, snapshot_date=snapshot_date)
    merged = features.merge(rfm, on=CUSTOMER_ID_COL, how="left")
    return merged.sort_values(CUSTOMER_ID_COL).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Preprocessing pipeline
# -----------------------------------------------------------------------------


def build_preprocessing_pipeline(
    numeric_features: Sequence[str] = NUMERIC_FEATURES,
    woe_features: Sequence[str] = WOE_FEATURES,
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES,
) -> Pipeline:
    """Construct the preprocessing pipeline used for model training."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                list(dict.fromkeys(list(numeric_features) + list(woe_features))),
            ),
            (
                "categorical",
                categorical_pipeline,
                list(categorical_features),
            ),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("woe", WoEEncoder(woe_features)),
            ("preprocess", preprocessor),
        ]
    )

    return pipeline


def get_feature_names(preprocessor: ColumnTransformer) -> Sequence[str]:
    """Return feature names after transformation for reporting purposes."""

    feature_names: list[str] = []

    numeric_names = preprocessor.transformers_[0][2]
    feature_names.extend(numeric_names)

    categorical_encoder: OneHotEncoder = preprocessor.transformers_[1][1].named_steps["encoder"]
    categorical_feature_names = list(categorical_encoder.get_feature_names_out(preprocessor.transformers_[1][2]))
    feature_names.extend(categorical_feature_names)

    return feature_names


def build_baseline_dataframe(df: pd.DataFrame, limit: Optional[int] = None) -> pd.DataFrame:
    """Retain compatibility with the interim EDA helper."""

    return df.head(limit).copy() if limit is not None else df.copy()
