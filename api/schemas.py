from typing import List, Optional
from pydantic import BaseModel


class PredictionResponse(BaseModel):
    success: bool
    nominal: Optional[str] = None
    label: Optional[str] = None
    confidence: Optional[float] = None
    is_confident: Optional[bool] = None
    all_probabilities: Optional[dict] = None
    message: Optional[str] = None


class BatchPredictionItem(PredictionResponse):
    filename: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    success: bool
    total: int
    results: List[BatchPredictionItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    coin_model_loaded: Optional[bool] = None
    model_name: str
    classes: list
    threshold: float
