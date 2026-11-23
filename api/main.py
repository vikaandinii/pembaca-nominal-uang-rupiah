from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api import predictor
from api.schemas import (
    BatchPredictionItem,
    BatchPredictionResponse,
    HealthResponse,
    PredictionResponse,
)

ALLOWED_IMAGE_TYPES = ("image/jpeg", "image/png", "image/webp")
model_store = {"model": None, "coin_pt_model": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_store["model"] = predictor.load_model()
    model_store["coin_pt_model"] = predictor.load_coin_pt_model()
    yield


app = FastAPI(
    title="Rupiah Detector API",
    description="API klasifikasi uang rupiah (kertas & koin) dengan arsitektur MobileNetV2",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/", tags=["Info"])
def root():
    return {
        "message": "Rupiah Detector API is running",
        "model": "MobileNetV2",
        "docs": "/docs",
        "health": "/health",
        "endpoints": ["/predict", "/predict/batch", "/coin", "/coin/batch"],
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health():
    model_loaded = model_store["model"] is not None
    return HealthResponse(
        status="ok" if model_loaded else "model_not_loaded",
        model_loaded=model_loaded,
        coin_model_loaded=model_store["coin_pt_model"] is not None,
        model_name="MobileNetV2",
        classes=predictor.CLASSES,
        threshold=predictor.CONFIDENCE_THRESHOLD,
    )


def _get_model(key: str, error_detail: str):
    model = model_store.get(key)
    if model is None:
        raise HTTPException(status_code=503, detail=error_detail)
    return model


async def _predict_upload(file: UploadFile, predict_fn, model) -> dict:
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Format file ga didukung ({file.filename}): {file.content_type}",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail=f"File kosong: {file.filename}")

    try:
        result = predict_fn(model, image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal proses gambar ({file.filename}): {e}")

    return {
        **result,
        "message": None if result["is_confident"] else "Gambar kurang jelas, coba foto ulang ya.",
    }


async def _predict_batch(files: List[UploadFile], predict_fn, model) -> BatchPredictionResponse:
    results = []
    for file in files:
        res = await _predict_upload(file, predict_fn, model)
        results.append(BatchPredictionItem(filename=file.filename, success=True, **res))
    return BatchPredictionResponse(success=True, total=len(results), results=results)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediksi Uang Kertas"])
async def predict(file: UploadFile = File(...)):
    model = _get_model("model", "Model uang kertas belum siap, coba ditraining dulu.")
    return PredictionResponse(success=True, **(await _predict_upload(file, predictor.predict, model)))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediksi Uang Kertas"])
async def predict_batch(files: List[UploadFile] = File(...)):
    model = _get_model("model", "Model uang kertas belum siap, coba ditraining dulu.")
    return await _predict_batch(files, predictor.predict, model)


@app.post("/coin", response_model=PredictionResponse, tags=["Prediksi Koin"])
async def predict_coin(file: UploadFile = File(...)):
    model = _get_model("coin_pt_model", "Model koin PyTorch (best_mobilenetv2.pt) belum siap.")
    return PredictionResponse(success=True, **(await _predict_upload(file, predictor.predict_coin_pt, model)))


@app.post("/coin/batch", response_model=BatchPredictionResponse, tags=["Prediksi Koin"])
async def predict_coin_batch(files: List[UploadFile] = File(...)):
    model = _get_model("coin_pt_model", "Model koin PyTorch (best_mobilenetv2.pt) belum siap.")
    return await _predict_batch(files, predictor.predict_coin_pt, model)
