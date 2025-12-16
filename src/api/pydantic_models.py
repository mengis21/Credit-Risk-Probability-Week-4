"""Request and response schemas for the FastAPI service."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class TransactionRecord(BaseModel):
    TransactionId: str
    CustomerId: str
    Amount: float
    Value: float
    TransactionStartTime: datetime
    ProductCategory: str
    ChannelId: str
    ProviderId: str
    PricingStrategy: int = 0
    BatchId: Optional[str] = None
    AccountId: Optional[str] = None
    SubscriptionId: Optional[str] = None
    CurrencyCode: Optional[str] = Field(default="UGX")
    CountryCode: Optional[str] = Field(default="256")
    ProductId: Optional[str] = None


class PredictionRequest(BaseModel):
    transactions: List[TransactionRecord]
    snapshot_date: Optional[datetime] = Field(
        default=None,
        description="Override the snapshot date used for recency calculations.",
    )


class PredictionItem(BaseModel):
    customer_id: str
    probability: float


class PredictionResponse(BaseModel):
    results: List[PredictionItem]
