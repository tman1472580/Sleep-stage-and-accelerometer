#!/usr/bin/env python3
"""
Real-time sleep scoring server for Polar H10 accelerometer data.
- Accepts streaming accel samples from the browser
- Buffers and scores using a U-Time model
- Adds robust CORS for http://localhost:8000 and http://127.0.0.1:8000
- Avoids score() reshape pitfalls by calling model.predict() directly
"""

from flask import Flask, request, jsonify

import numpy as np
import pandas as pd
from pathlib import Path
import threading
import time
from collections import deque

# ---- zmax-datasets / model imports ----
from zmax_datasets.utils.data import Data
from zmax_datasets.transforms.accelerometer import ProcessDynamicAccelerometer
from zmax_datasets.processing.sleep_scoring import UTimeModel

ALLOWED_ORIGINS = {
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:5500",  # Add common dev server ports
    "http://127.0.0.1:5500",
    "http://localhost:3000",  # Add more as needed
    "http://127.0.0.1:3000",
}


from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins (dev only!)

@app.after_request
def add_cors_headers(resp):
    origin = request.headers.get("Origin", "")
    if origin in ALLOWED_ORIGINS:
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Vary"] = "Origin"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return resp

# ========= Configuration =========
EPOCH_LENGTH_S = 30           # seconds
SAMPLING_RATE = 200           # Hz
SAMPLES_PER_EPOCH = EPOCH_LENGTH_S * SAMPLING_RATE  # 6000

# From your model logs:
MODEL_SAMPLES_PER_PERIOD = 1920  # samples per period
MODEL_PERIODS_PER_BATCH = 11     # periods per batch
TARGET_TOTAL = MODEL_SAMPLES_PER_PERIOD * MODEL_PERIODS_PER_BATCH  # 21120

# Buffer ~ 2x the required duration (safe margin)
BUFFER_DURATION_S = max(180, (TARGET_TOTAL / SAMPLING_RATE) * 2)
BUFFER_SIZE = int(BUFFER_DURATION_S * SAMPLING_RATE)

LABELS = np.array(["Wake", "NREM", "REM"], dtype=object)

print(f"[config] Model wants {MODEL_PERIODS_PER_BATCH}x{MODEL_SAMPLES_PER_PERIOD} = {TARGET_TOTAL} samples")
print(f"[config] Buffer size: {BUFFER_DURATION_S:.1f}s ({BUFFER_SIZE} samples)")

# ========= State =========
class SleepScorer:
    def __init__(self, weights_path: str):
        self.model = None
        self.buffer = deque(maxlen=BUFFER_SIZE)   # rolling sample buffer of dicts with t,x,y,z
        self.lock = threading.Lock()
        self.predictions = []
        self.last_prediction_time = 0.0

        self._load_model(weights_path)

    def _load_model(self, weights_path: str):
        weights_path = Path(weights_path).expanduser().resolve()
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")

        # UTimeModel expects the experiment root (contains hyperparameters/, model/, etc.)
        experiment_root = weights_path.parent.parent  # .../models
        hparams_path = experiment_root / "hyperparameters" / "hparams.yaml"
        if not hparams_path.exists():
            raise FileNotFoundError(f"Missing hparams: {hparams_path}")

        print(f"[init] Loading model from: {experiment_root}")
        self.model = UTimeModel(str(experiment_root))
        # load keras weights
        self.model.model.load_weights(str(weights_path))
        print("[init] Model loaded successfully")

    # ---- buffer ----
    def add_samples(self, samples):
        with self.lock:
            for s in samples:
                self.buffer.append({
                    "t": float(s["t"]),
                    "x": float(s["x"]),
                    "y": float(s["y"]),
                    "z": float(s["z"]),
                })

    def _current_window_df(self):
        """Return a DataFrame with exactly TARGET_TOTAL most-recent samples, or None."""
        if len(self.buffer) < TARGET_TOTAL:
            return None
        df = pd.DataFrame(self.buffer)
        df = df.sort_values("t")
        if len(df) > TARGET_TOTAL:
            df = df.tail(TARGET_TOTAL)
        if len(df) < TARGET_TOTAL:
            return None
        return df

    def get_collection_status(self):
        with self.lock:
            n = len(self.buffer)
            required = TARGET_TOTAL
            if n == 0:
                return dict(ready=False, samples=0, required=required,
                            duration_s=0.0,
                            required_duration_s=required / SAMPLING_RATE,
                            progress=0.0)
            t = [s["t"] for s in self.buffer]
            duration = max(t) - min(t)
            progress = min(100.0, (n / required) * 100.0)
            return dict(
                ready=n >= required,
                samples=n,
                required=required,
                duration_s=duration,
                required_duration_s=required / SAMPLING_RATE,
                progress=progress,
            )

    def get_latest_prediction(self):
        with self.lock:
            return self.predictions[-1] if self.predictions else None

    def get_all_predictions(self):
        with self.lock:
            return list(self.predictions)

    # ---- scoring ----
    def score_now(self):
        """Score the most recent TARGET_TOTAL samples. Returns a prediction dict or None."""
        with self.lock:
            df = self._current_window_df()
            if df is None:
                return None

            # Prepare Data -> Transform (zmax pipeline)
            accel_xyz = df[["x", "y", "z"]].to_numpy()
            data = Data(np.asarray(accel_xyz), sample_rate=SAMPLING_RATE)

            # Some versions of ProcessDynamicAccelerometer are callables/classes — try both
            try:
                transformer = ProcessDynamicAccelerometer()
            except TypeError:
                transformer = ProcessDynamicAccelerometer

            sample = None
            for kwargs in [
                {"epoch_length": EPOCH_LENGTH_S, "sampling_frequency": SAMPLING_RATE},
                {"epoch_length": EPOCH_LENGTH_S, "fs": SAMPLING_RATE},
                {},
            ]:
                try:
                    sample = transformer(data, **kwargs)
                    break
                except TypeError:
                    continue

            if sample is None or not hasattr(sample, "array"):
                print("[error] Transform failed, no .array on returned sample")
                return None

            arr = np.asarray(sample.array)  # shape (N, C) expected
            if arr.ndim != 2:
                print(f"[error] Unexpected transformed shape: {arr.shape}")
                return None

            # Make sure we pass exactly one batch shaped (1, 11, 1920, 2) into Keras model
            n, c = arr.shape
            if c <= 0:
                print(f"[error] Invalid channel count in transformed array: {arr.shape}")
                return None

            if n < TARGET_TOTAL:
                # not enough yet (shouldn’t happen because we sliced df already)
                return None

            arr_win = arr[-TARGET_TOTAL:]  # last exact window
            try:
                x = arr_win.reshape(1, MODEL_PERIODS_PER_BATCH, MODEL_SAMPLES_PER_PERIOD, c).astype(np.float32)
            except Exception as e:
                print(f"[error] reshape to (1,{MODEL_PERIODS_PER_BATCH},{MODEL_SAMPLES_PER_PERIOD},{c}) failed: {e}")
                return None

            # Predict (no reliance on score()/prepare_data() that was dropping to zero)
            try:
                y = self.model.model.predict(x, verbose=0)
            except Exception as e:
                print(f"[error] model.predict failed: {e}")
                return None

            # Normalize output shapes:
            # Possibilities:
            #   (1, 11, n_classes)  -> take last period
            #   (1, n_classes)      -> probabilities for the batch
            #   (11, n_classes)     -> take last
            #   (n_classes,)        -> as-is
            probs = None
            if isinstance(y, (list, tuple)):
                y = y[0]
            y = np.asarray(y)

            if y.ndim == 3 and y.shape[0] == 1:
                probs = y[0, -1]  # (n_classes,)
            elif y.ndim == 2:
                if y.shape[0] == 1:
                    probs = y[0]
                elif y.shape[0] == MODEL_PERIODS_PER_BATCH:
                    probs = y[-1]
                else:
                    probs = y[-1]
            elif y.ndim == 1:
                probs = y
            else:
                print(f"[warning] unexpected model output shape {y.shape}, using argmax on last axis if possible")
                try:
                    probs = y.reshape(-1)[-3:]  # naïve fallback
                except Exception:
                    return None

            probs = np.asarray(probs, dtype=np.float32)
            if probs.ndim != 1:
                probs = probs.ravel()
            # Map to up to 3 labels
            labels = LABELS[: len(probs)]
            # Softmax if not already probabilities
            if probs.min() < 0 or probs.max() > 1.0 or abs(probs.sum() - 1.0) > 1e-3:
                e = np.exp(probs - probs.max())
                probs = e / (e.sum() + 1e-9)
            pred_idx = int(np.argmax(probs))

            min_t = float(df["t"].min())
            max_t = float(df["t"].max())
            pred = dict(
                epoch_start=min_t,
                epoch_end=max_t,
                label=str(labels[pred_idx]),
                probabilities={str(labels[i]): float(probs[i]) for i in range(len(labels))},
                timestamp=time.time(),
                samples_used=int(len(df)),
            )

            self.predictions.append(pred)
            self.last_prediction_time = max_t
            print(f"[score] {pred['label']}  (probs={pred['probabilities']})")
            return pred

# ---- Initialize scorer ----
WEIGHTS_PATH = "./models/model/model_weights.h5"
scorer = None
try:
    scorer = SleepScorer(WEIGHTS_PATH)
except Exception as e:
    print(f"[error] Failed to load model: {e}")
    scorer = None

# ========= Routes =========
def _preflight():
    return ("", 204)

@app.route("/health", methods=["GET", "OPTIONS"])
def health():
    if request.method == "OPTIONS":
        return _preflight()
    return jsonify({
        "status": "ok",
        "model_loaded": scorer is not None,
        "config": {
            "required_duration_s": TARGET_TOTAL / SAMPLING_RATE,
            "required_samples": TARGET_TOTAL,
            "model_periods": MODEL_PERIODS_PER_BATCH,
            "samples_per_period": MODEL_SAMPLES_PER_PERIOD,
        },
    })

@app.route("/api/status", methods=["GET", "OPTIONS"])
def api_status():
    if request.method == "OPTIONS":
        return _preflight()
    if scorer is None:
        return jsonify({"error": "Model not loaded"}), 500
    return jsonify(scorer.get_collection_status())

@app.route("/api/samples", methods=["POST", "OPTIONS"])
def api_samples():
    if request.method == "OPTIONS":
        return _preflight()
    if scorer is None:
        return jsonify({"error": "Model not loaded"}), 500
    data = request.get_json(silent=True) or {}
    if "samples" not in data:
        return jsonify({"error": "Invalid request format"}), 400
    scorer.add_samples(data["samples"])

    status = scorer.get_collection_status()
    pred = None
    # score every ~period length once we have enough samples
    try:
        if status["ready"]:
            pred = scorer.score_now()
    except Exception as e:
        print(f"[error] scoring step failed: {e}")

    return jsonify({
        "received": len(data["samples"]),
        "buffer_size": len(scorer.buffer),
        "status": status,
        "prediction": pred,
    })

@app.route("/api/latest", methods=["GET", "OPTIONS"])
def api_latest():
    if request.method == "OPTIONS":
        return _preflight()
    if scorer is None:
        return jsonify({"error": "Model not loaded"}), 500
    pred = scorer.get_latest_prediction()
    status = scorer.get_collection_status()
    if pred is None:
        msg = ("Ready to predict - waiting for scoring interval"
               if status.get("ready") else
               f"Collecting data... {status.get('progress', 0):.0f}% "
               f"({status.get('duration_s', 0):.0f}s / {status.get('required_duration_s', 0):.0f}s needed)")
        return jsonify({"status": "no_prediction", "message": msg, "collection_status": status})
    return jsonify({"status": "ok", "prediction": pred, "collection_status": status})

@app.route("/api/history", methods=["GET", "OPTIONS"])
def api_history():
    if request.method == "OPTIONS":
        return _preflight()
    if scorer is None:
        return jsonify({"error": "Model not loaded"}), 500
    preds = scorer.get_all_predictions()
    return jsonify({"status": "ok", "count": len(preds), "predictions": preds})

@app.route("/api/reset", methods=["POST", "OPTIONS"])
def api_reset():
    if request.method == "OPTIONS":
        return _preflight()
    if scorer is None:
        return jsonify({"error": "Model not loaded"}), 500
    with scorer.lock:
        scorer.buffer.clear()
        scorer.predictions.clear()
        scorer.last_prediction_time = 0.0
    return jsonify({"status": "ok", "message": "Scorer reset"})

if __name__ == "__main__":
    print("[start] Flask on http://127.0.0.1:5000 (debug)")
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)
