# Rupiah Detector

Klasifikasi uang kertas rupiah (7 kelas) menggunakan MobileNetV2.

## Kelas Nominal
`1000`, `2000`, `5000`, `10000`, `20000`, `50000`, `100000`

## Struktur Folder
```
rupiah-detector/
├── api/
│   ├── main.py          # FastAPI server
│   ├── predictor.py     # load model & prediksi
│   └── schemas.py       # response schema
├── notebooks/
│   └── mobilenetv2_training.ipynb
├── models/
├── dataset/
└── requirements.txt
```

## Cara Menjalankan API
```bash
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

Dokumentasi API: http://localhost:8000/docs
