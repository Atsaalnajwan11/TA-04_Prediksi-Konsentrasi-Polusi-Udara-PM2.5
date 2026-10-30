"""Aplikasi Flask: Peramalan & Proyeksi PM2.5 Beijing dengan LSTM (TA-04).

Mode peramalan:
  1. "data"      : waktu mulai berada di dalam / tepat setelah data (<= 2015-01-01 00:00).
                   Jendela 24 jam diambil dari data nyata.
  2. "proyeksi"  : waktu mulai di masa depan yang jauh (2015-01-01 01:00 dst., mis. 2026).
                   Jendela 24 jam dan variabel cuaca dibentuk dari klimatologi 2010-2014
                   (rata-rata per hari-dalam-tahun & jam, dihaluskan). Hasilnya adalah pola
                   musiman yang dipelajari model, BUKAN prakiraan kejadian nyata.
"""
import os, json
import numpy as np, pandas as pd
from flask import Flask, jsonify, render_template, request

BASE = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

# ---------- Inferensi LSTM murni NumPy (tanpa TensorFlow, cocok untuk Vercel) ----------
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class NumpyLSTM:
    """Replika forward-pass model Keras: LSTM(64, seq) -> LSTM(32) -> Dense(16, relu) -> Dense(1).
    Dropout tidak dipakai saat inferensi. Urutan gate Keras: input, forget, cell, output."""

    def __init__(self, path):
        z = np.load(path)
        self.w = {k: z[k].astype(np.float64) for k in z.files}

    @staticmethod
    def _lstm(x, K, R, b, return_seq):
        B, T_, _ = x.shape
        U = R.shape[0]
        h, c = np.zeros((B, U)), np.zeros((B, U))
        xs = x @ K + b                                              # (B, T, 4U)
        out = []
        for t in range(T_):
            i, f, g, o = np.split(xs[:, t] + h @ R, 4, axis=1)
            i, f, o, g = _sigmoid(i), _sigmoid(f), _sigmoid(o), np.tanh(g)
            c = f * c + i * g
            h = o * np.tanh(c)
            out.append(h)
        return np.stack(out, axis=1) if return_seq else h

    def predict(self, X, batch_size=2048):
        X = np.asarray(X, dtype=np.float64)
        w, res = self.w, []
        for s in range(0, len(X), batch_size):
            h = self._lstm(X[s:s + batch_size], w["lstm1_kernel"], w["lstm1_recurrent"], w["lstm1_bias"], True)
            h = self._lstm(h, w["lstm2_kernel"], w["lstm2_recurrent"], w["lstm2_bias"], False)
            h = np.maximum(h @ w["dense1_kernel"] + w["dense1_bias"], 0.0)
            res.append((h @ w["dense2_kernel"] + w["dense2_bias"]).reshape(-1))
        return np.concatenate(res)


# ---------- Muat artefak ----------
META = json.load(open(os.path.join(BASE, "model", "meta.json")))
_SC = json.load(open(os.path.join(BASE, "model", "scaler.json")))
SC_SCALE, SC_MIN = np.array(_SC["scale"]), np.array(_SC["min"])
MODEL = NumpyLSTM(os.path.join(BASE, "model", "weights.npz"))
DF = pd.read_csv(os.path.join(BASE, "data", "pm25_clean.csv"), parse_dates=["datetime"], index_col="datetime")


def scale(a):
    """Setara MinMaxScaler.transform: x * scale_ + min_."""
    return np.asarray(a, dtype=float) * SC_SCALE + SC_MIN


FEATURES, T = META["features"], META["timesteps"]
DF = DF[FEATURES]
D_MIN, D_SCALE = _SC["data_min"][0], _SC["scale"][0]
SCALED = scale(DF.values)
POS = {ts: i for i, ts in enumerate(DF.index)}
NF = len(FEATURES)
MAX_YEAR = 2100
_cache = {}

# ---------- Klimatologi (untuk proyeksi masa depan) ----------
_CUM = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])


def doy_idx(month, day):
    """Hari-dalam-tahun 0..364 (kalender non-kabisat; 29 Feb dipetakan ke 28 Feb)."""
    month, day = np.asarray(month), np.asarray(day)
    d = _CUM[month - 1] + day - 1
    return np.where((month == 2) & (day == 29), 58, d)


def build_climatology(win=15):
    d = DF.copy()
    d["doy"] = doy_idx(d.index.month, d.index.day)
    d["hr"] = d.index.hour
    g = d.groupby(["doy", "hr"])[FEATURES].mean()
    full = pd.MultiIndex.from_product([range(365), range(24)], names=["doy", "hr"])
    clim = g.reindex(full).interpolate().to_numpy().reshape(365, 24, NF)
    pad, k = win // 2, np.ones(win) / win                      # smoothing sirkular antar hari
    ext = np.concatenate([clim[-pad:], clim, clim[:pad]], axis=0)
    sm = np.apply_along_axis(lambda x: np.convolve(x, k, mode="valid"), 0, ext)
    return scale(sm.reshape(-1, NF)).reshape(365, 24, NF)


CLIM_S = build_climatology()


def clim_rows(times):
    t = pd.DatetimeIndex(times)
    return CLIM_S[doy_idx(t.month, t.day), t.hour].copy()


def clim_window(ts):
    return clim_rows(ts - pd.to_timedelta(np.arange(T, 0, -1), unit="h"))


# ---------- Util ----------
def inv(a):
    return np.clip(np.asarray(a, dtype=float).reshape(-1) / D_SCALE + D_MIN, 0, None)


def category(v):
    if v <= 35:  return {"label": "Baik", "color": "#15803d"}
    if v <= 75:  return {"label": "Sedang", "color": "#a16207"}
    if v <= 150: return {"label": "Tidak sehat", "color": "#c2410c"}
    return {"label": "Sangat tidak sehat", "color": "#b91c1c"}


def parse_ts(s):
    try:
        return pd.Timestamp(s).floor("h")
    except Exception:
        return None


def forecast_data(window_scaled, steps):
    """Rekursif dari data nyata; fitur cuaca dipertahankan (persisten)."""
    w, out = window_scaled.copy(), []
    for _ in range(steps):
        p = float(MODEL.predict(w[None, :, :])[0])
        out.append(p)
        nxt = w[-1].copy(); nxt[0] = p
        w = np.vstack([w[1:], nxt])
    return inv(out)


def forecast_clim(ts, steps):
    """Rekursif dari jendela klimatologi; fitur cuaca tiap jam = klimatologi jam itu."""
    w = clim_window(ts)
    times = pd.date_range(ts, periods=steps, freq="h")
    exo, out = clim_rows(times), []
    for s in range(steps):
        p = float(MODEL.predict(w[None, :, :])[0])
        out.append(p)
        nxt = exo[s]; nxt[0] = p
        w = np.vstack([w[1:], nxt])
    return inv(out)


# ---------- Halaman ----------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return {"status": "ok"}


# ---------- API ----------
@app.route("/api/info")
def info():
    last = DF.index[-1]
    return jsonify(
        start=str(DF.index.min()), end=str(last), n=len(DF), timesteps=T, features=FEATURES,
        metrics=META["metrics"], split=META["split"], mean=float(DF["pm2.5"].mean()), max=float(DF["pm2.5"].max()),
        min_forecast=str(DF.index[T]), max_forecast=f"{MAX_YEAR}-12-31 23:00", first_future_year=int(last.year) + 1, max_year=MAX_YEAR,
    )


@app.route("/api/historical")
def historical():
    res = request.args.get("resample", "D")
    s, e = parse_ts(request.args.get("start", DF.index.min())), parse_ts(request.args.get("end", DF.index.max()))
    d = DF.loc[s:e, "pm2.5"]
    if res != "H":
        d = d.resample({"D": "D", "W": "W", "M": "MS"}.get(res, "D")).mean()
    d = d.round(1)
    return jsonify(labels=[t.strftime("%Y-%m-%d %H:%M") for t in d.index], values=d.tolist())


@app.route("/api/test_predictions")
def test_predictions():
    if "test" not in _cache:
        i0 = POS[pd.Timestamp(META["split"]["test_start"])]
        X = np.stack([SCALED[i - T:i] for i in range(i0, len(DF))])
        _cache["test"] = pd.Series(inv(MODEL.predict(X)), index=DF.index[i0:])
    pred = _cache["test"]
    start = parse_ts(request.args.get("start", pred.index[0]))
    days = min(int(request.args.get("days", 7)), 60)
    sl = pred.loc[start:start + pd.Timedelta(days=days) - pd.Timedelta(hours=1)]
    act = DF.loc[sl.index, "pm2.5"]
    return jsonify(labels=[t.strftime("%Y-%m-%d %H:%M") for t in sl.index],
                   actual=act.round(1).tolist(), predicted=sl.round(1).tolist(),
                   range=[str(pred.index[0]), str(pred.index[-1])])


@app.route("/api/forecast")
def api_forecast():
    """Ramalan 1..48 jam mulai 'start'. Data nyata bila start <= 2015-01-01 00:00, selain itu proyeksi klimatologi."""
    start = parse_ts(request.args.get("start"))
    try:
        steps = max(1, min(int(request.args.get("steps", 24)), 48))
    except ValueError:
        return jsonify(error="Parameter steps harus berupa angka."), 400
    last = DF.index[-1]
    if start is None:
        return jsonify(error="Format waktu tidak valid. Contoh: 2026-01-15T08:00."), 400
    if start < DF.index[T]:
        return jsonify(error=f"Waktu terlalu awal. Pilih mulai {DF.index[T]:%d-%m-%Y %H:%M}."), 400
    if start.year > MAX_YEAR:
        return jsonify(error=f"Tahun maksimum yang didukung adalah {MAX_YEAR}."), 400

    if start <= last + pd.Timedelta(hours=1):
        mode, i = "data", POS.get(start, len(DF))
        pred = forecast_data(SCALED[i - T:i], steps)
        input_last = round(float(inv(SCALED[i - 1, 0])[0]), 1)
    else:
        mode, input_last = "proyeksi", None
        pred = forecast_clim(start, steps)

    times = pd.date_range(start, periods=steps, freq="h")
    actual = [float(DF["pm2.5"].get(t, np.nan)) for t in times]
    actual = [None if np.isnan(a) else round(a, 1) for a in actual]
    valid = [(a, p) for a, p in zip(actual, pred) if a is not None]
    mae = float(np.mean([abs(a - p) for a, p in valid])) if valid else None
    return jsonify(mode=mode, labels=[t.strftime("%Y-%m-%d %H:%M") for t in times], predicted=np.round(pred, 1).tolist(),
                   actual=actual, mae=None if mae is None else round(mae, 2), next_hour=round(float(pred[0]), 1),
                   category=category(pred[0]), mean_pred=round(float(pred.mean()), 1), max_pred=round(float(pred.max()), 1),
                   input_last=input_last)


@app.route("/api/projection_year")
def projection_year():
    """Proyeksi harian PM2.5 untuk satu tahun di masa depan (pola musiman yang dipelajari LSTM)."""
    try:
        year = int(request.args.get("year", 2026))
    except ValueError:
        return jsonify(error="Tahun harus berupa angka, mis. 2026."), 400
    first = int(DF.index[-1].year) + 1
    if not first <= year <= MAX_YEAR:
        return jsonify(error=f"Masukkan tahun {first}–{MAX_YEAR}. Untuk 2010–2014 gunakan bagian data historis."), 400
    if year in _cache:
        return jsonify(_cache[year])

    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    W = np.stack([clim_window(d) for d in days])                       # (B, T, F)
    doy = doy_idx(days.month, days.day)
    preds = []
    for h in range(24):                                                  # 24 jam per hari, semua hari sekaligus
        p = MODEL.predict(W).reshape(-1)
        row = CLIM_S[doy, h].copy(); row[:, 0] = p
        W = np.concatenate([W[:, 1:], row[:, None, :]], axis=1)
        preds.append(p)
    P = inv(np.stack(preds, axis=1)).reshape(len(days), 24)             # PM2.5 per jam per hari
    daily = P.mean(axis=1)
    s = pd.Series(daily, index=days)
    monthly = s.resample("MS").mean()
    hist_month = DF["pm2.5"].groupby(DF.index.month).mean()
    out = dict(
        year=year,
        labels=[d.strftime("%Y-%m-%d") for d in days], daily=np.round(daily, 1).tolist(),
        monthly=np.round(monthly.values, 1).tolist(),
        monthly_hist=np.round(hist_month.reindex(range(1, 13)).values, 1).tolist(),
        month_names=["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"],
        annual_mean=round(float(daily.mean()), 1), peak_month=int(monthly.idxmax().month), low_month=int(monthly.idxmin().month),
        peak_value=round(float(monthly.max()), 1), low_value=round(float(monthly.min()), 1),
        days_over_75=int((daily > 75).sum()), n_days=len(days),
        category=category(float(daily.mean())),
    )
    _cache[year] = out
    return jsonify(out)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)