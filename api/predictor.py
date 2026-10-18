import numpy as np
from PIL import Image
import io
import os

# 7 kelas nominal rupiah
CLASSES = ["1000", "2000", "5000", "10000", "20000", "50000", "100000"]
IMG_SIZE = (224, 224)

# batas minimal confidence, kalau di bawah ini dianggap ragu
CONFIDENCE_THRESHOLD = 0.60

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "mobilenetv2_rupiah.keras")


def load_model():
    # load model pas server baru nyala
    try:
        import tensorflow as tf
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"model berhasil dimuat dari {MODEL_PATH}")
            return model
        else:
            print(f"model belum ada di {MODEL_PATH}, ditraining dulu ya")
            return None
    except Exception as e:
        print(f"error pas load model: {e}")
        return None


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    # sesuaikan ukuran gambar ke 224x224 dan normalisasi pixel ke 0-1
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict(model, image_bytes: bytes) -> dict:
    arr = preprocess_image(image_bytes)
    probs = model.predict(arr, verbose=0)[0]

    pred_idx = int(np.argmax(probs))
    pred_class = CLASSES[pred_idx]
    confidence = float(probs[pred_idx])

    # simpen semua nilai probabilitas buat dikirim ke frontend
    all_probs = {cls: round(float(p), 4) for cls, p in zip(CLASSES, probs)}

    return {
        "nominal": pred_class,
        "label": f"Rp {int(pred_class):,}".replace(",", "."),
        "confidence": round(confidence, 4),
        "is_confident": confidence >= CONFIDENCE_THRESHOLD,
        "all_probabilities": all_probs,
    }
