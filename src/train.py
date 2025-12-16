"""Model training and MLflow tracking orchestration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import mlflow
import mlflow.sklearn  # noqa: F401 - ensures sklearn flavor is registered
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from .data_processing import (
    CUSTOMER_ID_COL,
    build_preprocessing_pipeline,
    build_training_dataset,
    get_feature_names,
    load_transactions,
)


MLFLOW_TRACKING_URI = "file:mlruns"
MLFLOW_EXPERIMENT = "credit-risk-probability"
REGISTERED_MODEL_NAME = "credit-risk-probability-model"
ARTIFACTS_DIR = Path("models")
TRAINING_OUTPUT = Path("data/processed/model_training_dataset.parquet")
METADATA_OUTPUT = Path("data/processed/model_training_metadata.parquet")


def _evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    roc_auc = roc_auc_score(y_test, y_proba)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
    }


def train_models(data_path: Path, snapshot_date: str | None = None) -> None:
    raw_df = load_transactions(data_path)

    snapshot = pd.to_datetime(snapshot_date) if snapshot_date else None
    feature_df, target, metadata = build_training_dataset(raw_df, snapshot_date=snapshot)

    # Persist the model-ready dataset for reproducibility
    TRAINING_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    METADATA_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    feature_df.assign(is_high_risk=target.values).to_parquet(TRAINING_OUTPUT, index=False)
    metadata.to_parquet(METADATA_OUTPUT, index=False)

    X = feature_df.drop(columns=[CUSTOMER_ID_COL])
    y = target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42,
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    model_definitions = {
        "logistic_regression": {
            "estimator": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "param_grid": {
                "clf__C": [0.1, 1.0, 10.0],
                "clf__penalty": ["l2"],
                "clf__solver": ["lbfgs"],
            },
        },
        "random_forest": {
            "estimator": RandomForestClassifier(class_weight="balanced", random_state=42),
            "param_grid": {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [None, 10, 15],
                "clf__min_samples_leaf": [1, 3],
            },
        },
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, object]] = []

    for model_name, config in model_definitions.items():
        preprocess_pipeline = build_preprocessing_pipeline()
        modelling_pipeline = Pipeline(
            steps=[
                ("preprocess", preprocess_pipeline),
                ("clf", config["estimator"]),
            ]
        )

        search = GridSearchCV(
            modelling_pipeline,
            param_grid=config["param_grid"],
            scoring="f1",
            cv=3,
            n_jobs=-1,
        )

        with mlflow.start_run(run_name=model_name) as run:
            search.fit(X_train, y_train)
            best_pipeline: Pipeline = search.best_estimator_

            metrics = _evaluate_model(best_pipeline, X_test, y_test)
            metrics["train_f1_mean_cv"] = float(search.best_score_)

            mlflow.log_params({k: v for k, v in search.best_params_.items()})
            mlflow.log_metrics(metrics)

            feature_columns = get_feature_names(best_pipeline.named_steps["preprocess"].named_steps["preprocess"])
            feature_info_path = ARTIFACTS_DIR / f"{model_name}_feature_columns.json"
            feature_info_path.write_text(json.dumps(list(feature_columns), indent=2))
            mlflow.log_artifact(str(feature_info_path))

            mlflow.sklearn.log_model(best_pipeline, artifact_path="model")

            woe_info_path = ARTIFACTS_DIR / f"{model_name}_woe_information.json"
            woe_encoder = best_pipeline.named_steps["preprocess"].named_steps["woe"]
            woe_info_path.write_text(json.dumps(woe_encoder.information_values, indent=2))
            mlflow.log_artifact(str(woe_info_path))

            results.append(
                {
                    "model_name": model_name,
                    "metrics": metrics,
                    "run_id": run.info.run_id,
                }
            )

    if not results:
        raise RuntimeError("No models were trained.")

    best_result = max(results, key=lambda item: item["metrics"]["f1"])
    best_model_uri = f"runs:/{best_result['run_id']}/model"

    registration = mlflow.register_model(best_model_uri, REGISTERED_MODEL_NAME)
    client = MlflowClient()
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=registration.version,
        stage="Production",
        archive_existing_versions=True,
    )

    # Persist the production model locally for quick inspection
    production_model = mlflow.sklearn.load_model(best_model_uri)
    joblib.dump(production_model, ARTIFACTS_DIR / "latest_model.joblib")
    (ARTIFACTS_DIR / "latest_model_uri.txt").write_text(best_model_uri)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train credit risk models with MLflow logging.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/raw/data.csv"),
        help="Path to the raw transaction CSV file.",
    )
    parser.add_argument(
        "--snapshot-date",
        type=str,
        default=None,
        help="Optional snapshot date (YYYY-MM-DD) for RFM calculations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_models(args.data_path, snapshot_date=args.snapshot_date)


if __name__ == "__main__":
    main()
