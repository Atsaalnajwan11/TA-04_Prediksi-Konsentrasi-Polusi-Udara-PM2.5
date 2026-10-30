# TA-04 — Prediksi Konsentrasi PM2.5 Beijing dengan LSTM

Peramalan PM2.5 (µg/m³) satu jam ke depan dari 24 jam terakhir (PM2.5 + cuaca) dengan LSTM (TensorFlow/Keras) dan aplikasi web Flask.

| Metrik (data uji 2014) | LSTM | Baseline persistensi |
|---|---|---|
| RMSE | 20,73 | 21,91 |
| MAE | 11,84 | 11,82 |
| R² | 0,951 | 0,945 |

## Struktur
```
app.py                  # aplikasi Flask + API
templates/index.html    # antarmuka (Chart.js)
notebook/TA04_LSTM_PM25.ipynb   # EDA, preprocessing, training, evaluasi
model/                  # pm25_lstm.h5, pm25_lstm.keras, scaler.joblib, meta.json, metrics*.csv
data/                   # dataset asli + pm25_clean.csv (hasil pembersihan)
images/                 # grafik hasil notebook
Procfile, render.yaml, requirements.txt, .python-version
```

## Menjalankan lokal
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                                          # buka http://localhost:5000
```
Melatih ulang: jalankan seluruh sel `notebook/TA04_LSTM_PM25.ipynb` (≈ 6–8 menit di CPU); file di `model/` dan `data/pm25_clean.csv` akan diperbarui.

## Deploy (Render)
1. Push repo ke GitHub. 2. Render → New → Web Service → pilih repo (otomatis membaca `render.yaml`).
3. Build: `pip install -r requirements.txt`; Start: `gunicorn app:app --workers 1 --threads 2 --timeout 180 --bind 0.0.0.0:$PORT`.
Catatan: TensorFlow butuh RAM cukup besar (disarankan ≥ 1 GB). Gunakan versi library yang sama dengan `requirements.txt` agar file `.h5` terbaca benar.

## API
`/api/info` · `/api/historical?resample=D|W|M|H&start=&end=` · `/api/test_predictions?start=2014-03-01&days=7` · `/api/forecast?start=2014-03-01T08:00&steps=24`
