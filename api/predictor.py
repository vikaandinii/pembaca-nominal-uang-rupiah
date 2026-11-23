import io
import json
import os
import tempfile
import zipfile

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms

CLASSES = ["1000", "2000", "5000", "10000", "20000", "50000", "100000"]
COIN_CLASSES = ["100", "1000", "200", "500"]  # urutan sesuai sorted(label) di 03_mobilenetv2.ipynb
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.60

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "mobilenetv2_rupiah.keras")
COIN_MODEL_PATH = os.path.join(MODELS_DIR, "best_mobilenetv2.pt")

_COIN_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _strip_config_key(obj, key: str):
    if isinstance(obj, dict):
        obj.pop(key, None)
        for v in obj.values():
            _strip_config_key(v, key)
    elif isinstance(obj, list):
        for item in obj:
            _strip_config_key(item, key)


def _load_keras_compat(path: str):
    """Load model .keras dengan fallback jika config menyimpan argumen dari Keras versi lebih baru."""
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        if "quantization_config" not in str(e):
            raise

    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with zipfile.ZipFile(path, "r") as zin, zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == "config.json":
                    cfg = json.loads(data.decode("utf-8"))
                    _strip_config_key(cfg, "quantization_config")
                    data = json.dumps(cfg).encode("utf-8")
                zout.writestr(item, data)
        return tf.keras.models.load_model(tmp_path, compile=False)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def load_model():
    """Load model uang kertas (.keras)."""
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"[WARN] File model belum ada di {MODEL_PATH}")
            return None
        model = _load_keras_compat(MODEL_PATH)
        print(f"[OK] Model MobileNetV2 berhasil dimuat dari: {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"[ERROR] Gagal load model: {e}")
        return None


def load_coin_pt_model():
    """Load model koin PyTorch (.pt)."""
    try:
        if not os.path.exists(COIN_MODEL_PATH):
            print(f"[WARN] File model koin belum ada di {COIN_MODEL_PATH}")
            return None
        model = models.mobilenet_v2(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.classifier[1].in_features, len(COIN_CLASSES)),
        )
        model.load_state_dict(torch.load(COIN_MODEL_PATH, map_location="cpu", weights_only=True))
        model.eval()
        print(f"[OK] Model koin PyTorch berhasil dimuat dari: {COIN_MODEL_PATH}")
        return model
    except Exception as e:
        print(f"[ERROR] Gagal load model koin PyTorch: {e}")
        return None


def _open_rgb(image_bytes: bytes) -> Image.Image:
    with Image.open(io.BytesIO(image_bytes)) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def preprocess_image(model, image_bytes: bytes) -> np.ndarray:
    """Resize proporsional + padding hitam 224x224 dan normalisasi sesuai kebutuhan model Keras."""
    img = _open_rgb(image_bytes)
    resized = ImageOps.contain(img, IMG_SIZE, method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", IMG_SIZE, (0, 0, 0))
    canvas.paste(resized, ((IMG_SIZE[0] - resized.width) // 2, (IMG_SIZE[1] - resized.height) // 2))

    has_internal_rescaling = hasattr(model, "layers") and any(
        "rescale" in layer.name.lower() for layer in model.layers
    )
    arr = np.array(canvas, dtype=np.float32)
    if not has_internal_rescaling:
        arr = (arr / 127.5) - 1.0
    return np.expand_dims(arr, axis=0)


def _format_result(probs: np.ndarray, classes: list, tag: str) -> dict:
    pred_idx = int(np.argmax(probs))
    pred_class = classes[pred_idx]
    confidence = float(probs[pred_idx])
    all_probs = {cls: round(float(p), 4) for cls, p in zip(classes, probs)}
    print(f"[{tag}] Hasil Prediksi: Rp {int(pred_class):,} (Keyakinan: {confidence:.2%})".replace(",", "."))
    return {
        "nominal": pred_class,
        "label": f"Rp {int(pred_class):,}".replace(",", "."),
        "confidence": round(confidence, 4),
        "is_confident": confidence >= CONFIDENCE_THRESHOLD,
        "all_probabilities": all_probs,
    }


def predict(model, image_bytes: bytes) -> dict:
    arr = preprocess_image(model, image_bytes)
    probs = model.predict(arr, verbose=0)[0]
    return _format_result(probs, CLASSES, "API Predict")


def predict_coin_pt(model, image_bytes: bytes) -> dict:
    tensor = _COIN_TRANSFORM(_open_rgb(image_bytes)).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    return _format_result(probs, COIN_CLASSES, "API Predict Koin")
