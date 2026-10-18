from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.schemas import PredictionResponse, HealthResponse
from api import predictor

model_store = {"model": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # pas api start langsung load model biar ga reload tiap ada request
    model_store["model"] = predictor.load_model()
    yield


app = FastAPI(
    title="Rupiah Detector API",
    description="API klasifikasi uang rupiah pake MobileNetV2",
    version="0.1.0",
    lifespan=lifespan,
)

# allow cors biar frontend bisa akses api dari browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/", tags=["Info"])
def root():
    return {
        "project": "Rupiah Detector",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health():
    return HealthResponse(
        status="ok" if model_store["model"] is not None else "model_not_loaded",
        model_loaded=model_store["model"] is not None,
        model_name="MobileNetV2 (Transfer Learning)",
        classes=predictor.CLASSES,
        threshold=predictor.CONFIDENCE_THRESHOLD,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediksi"])
async def predict(file: UploadFile = File(...)):
    # pastiin model udah siap
    if model_store["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model belum siap, coba ditraining dulu.",
        )

    # cek format gambar
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=400,
            detail=f"Format file ga didukung: {file.content_type}",
        )

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="File kosong")

    try:
        result = predictor.predict(model_store["model"], image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal proses gambar: {str(e)}")

    return PredictionResponse(
        success=True,
        **result,
        message=None if result["is_confident"] else "Gambar kurang jelas, coba foto ulang ya.",
    )
