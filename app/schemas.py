"""
Pydantic schemas used by the FastAPI application.
"""

from typing import List

from pydantic import BaseModel, Field


class CustomerData(BaseModel):

    age: int = Field(..., ge=18, le=100)

    job: str

    marital: str

    education: str

    default: str

    balance: int

    housing: str

    loan: str

    contact: str

    day: int = Field(..., ge=1, le=31)

    month: str

    campaign: int = Field(..., ge=1)

    pdays: int

    previous: int = Field(..., ge=0)

    poutcome: str


class PredictionResponse(BaseModel):

    prediction: str

    probability: float

    threshold: float


class CampaignInsightResponse(BaseModel):

    prediction: str

    probability: float

    threshold: float

    priority: str

    positive_signals: List[str]

    negative_signals: List[str]

    recommended_action: str

    ai_insight: str