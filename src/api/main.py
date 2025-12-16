"""FastAPI application exposing the credit risk model."""
from __future__ import annotations

import logging
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException

from ..data_processing import CUSTOMER_ID_COL
from ..predict import predict_probabilities
from .pydantic_models import PredictionItem, PredictionRequest, PredictionResponse


LOGGER = logging.getLogger(__name__)

app = FastAPI(title="Credit Risk Probability API", version="1.0.0")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if not request.transactions:
        raise HTTPException(status_code=400, detail="At least one transaction is required.")

    records = [tx.model_dump() for tx in request.transactions]
    transactions_df = pd.DataFrame(records)

    try:
        scored = predict_probabilities(transactions_df, snapshot_date=request.snapshot_date)
    except Exception as exc:  # pragma: no cover - defensive HTTP translation
        LOGGER.exception("Failed to score request")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    results: List[PredictionItem] = [
        PredictionItem(customer_id=str(row[CUSTOMER_ID_COL]), probability=float(row["probability"]))
        for _, row in scored.iterrows()
    ]

    return PredictionResponse(results=results)
