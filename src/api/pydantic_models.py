"""Request and response schemas for the FastAPI service."""
from pydantic import BaseModel


class PredictionRequest(BaseModel):
    account_id: str
    amount: float


class PredictionResponse(BaseModel):
    probability: float
