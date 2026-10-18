from pydantic import BaseModel
from typing import Optional


class PredictionResponse(BaseModel):
    success: bool
    nominal: Optional[str] = None          # contoh: "50000"
    label: Optional[str] = None            # format rupiah, misal "Rp 50.000"
    confidence: Optional[float] = None     # skor keyakinan model (0-1)
    is_confident: Optional[bool] = None    # true kalau lolos batas minimal
    all_probabilities: Optional[dict] = None  # probabilitas per nominal
    message: Optional[str] = None          # pesan kalau foto kurang jelas


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    classes: list
    threshold: float
