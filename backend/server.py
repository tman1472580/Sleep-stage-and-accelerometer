#!/usr/bin/env python3
"""
Real-time sleep scoring server for Polar H10 accelerometer data.

Key changes:
- CORS fixed (no manual OPTIONS; works with http://localhost:8000).
- First-prediction gate uses a small warm-up window (default 2 periods),
  not the entire model input window, so the UI won't stall around ~292s.
- Confidence threshold (default 0.6) returns "Unknown" for low-confidence.
- Labels use model.class_names if available; falls back to ["Wake","NREM","REM"].
"""

import os
import time
import threading
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

from zmax_datasets.utils.data import Data
from zmax_datasets.transforms.accelerometer import ProcessDynamicAccelerometer
from zmax_datasets.processing.sleep_scoring import UTimeModel, score

# ========= Flask & CORS =========
app = Flask(__name__)
# Allow all origins for local development; lock down in production if needed.
CORS(app)

@app.after_request
def after_request(response):
    # Helps with Chrome Private Network Access preflights from secure contexts.
    response.headers.add("Access-Control-Allow-Private-Network", "true")
    return response

# ========= Configuration =========
FRONTEND_SAMPLING_RATE = 200             # Hz - Polar H10 stream used by the frontend
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))
MIN_PERIODS_FOR_FIRST_PRED = int(os.getenv("MIN_PERIODS_FOR_FIRST_PRED", "2"))  # warm-up periods
BUFFER_DURATION_S = int(os.getenv("BUFFER_DURATION_S", "300"))  # seconds of raw 200 Hz data to keep
BUFFER_SIZE = BUFFER_DURATION_S * FRONTEND_SAMPLING_RATE

# ========= Sleep Scorer =========
class SleepScorer:
    def __init__(self, model_dir: str):
        self.model = None
        self.processor = None
        self.buffer = deque(maxlen=BUFFER_SIZE)
        self.lock = threading.Lock()
        self.predictions = []
        self.last_prediction_time = 0.0

        # Model/processing params
        self.model_input_shape = None
        self.model_sample_rate = None
        self.model_periods = None
        self.model_samples_per_period = None
        self.period_duration = None
        self.labels = np.array(["Wake", "NREM", "REM"], dtype=object)

        # Readiness gate (how many 30s periods before first scoring)
        self.min_periods_for_first_pred = MIN_PERIODS_FOR_FIRST_PRED
        self.target_total_samples = None  # computed after model load

        self._load_model(model_dir)

    def _load_model(self, model_dir: str):
        model_path = Path(model_dir).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        print(f"[init] Loading model from: {model_path}")
        self.processor = ProcessDynamicAccelerometer()
        self.model = UTimeModel(model_dir=str(model_path), n_samples_per_prediction=None)

        # Extract model parameters
        self.model_input_shape = self.model.input_shape          # e.g., (periods, samples_per_period, channels)
        self.model_sample_rate = self.model.input_sample_rate     # e.g., 64.0 Hz
        self.model_periods = self.model_input_shape[0]
        self.model_samples_per_period = self.model_input_shape[1]
        self.period_duration = float(self.model.period_duration)  # seconds per period (often 30.0)

        # Labels if provided by the model
        self.labels = np.array(
            getattr(self.model, "class_names", ["Wake", "NREM", "REM"]),
            dtype=object
        )

        # FIRST-PRED GATE: only require a small warm-up window (min_periods_for_first_pred)
        required_seconds = self.min_periods_for_first_pred * self.period_duration
        self.target_total_samples = int(np.ceil(required_seconds * FRONTEND_SAMPLING_RATE))

        print(f"[init] Model loaded")
        print(f"[init] input_shape={self.model_input_shape} | model_sr={self.model_sample_rate} Hz")
        print(f"[init] period_duration={self.period_duration}s | model_periods={self.model_periods}")
        print(f"[init] labels={list(self.labels)}")
        print(f"[init] warm-up gate: {self.min_periods_for_first_pred} periods "
              f"(~{required_seconds:.1f}s) → {self.target_total_samples} samples @ {FRONTEND_SAMPLING_RATE} Hz")
        print(f"[init] buffer: {BUFFER_DURATION_S}s ({BUFFER_SIZE} samples @ {FRONTEND_SAMPLING_RATE} Hz)")

    def add_samples(self, samples):
        """Add [{t, x, y, z}, ...] from frontend."""
        with self.lock:
            for s in samples:
                self.buffer.append({
                    "t": float(s.get("t", time.time())),
                    "x": float(s.get("x", 0.0)),
                    "y": float(s.get("y", 0.0)),
                    "z": float(s.get("z", 0.0)),
                })

    def _get_window_data(self):
        """Return (Data, t_min, t_max) when we have enough raw samples, else None."""
        required = self.target_total_samples or 0
        if len(self.buffer) < required:
            return None

        df = pd.DataFrame(list(self.buffer)).sort_values("t").tail(required)
        if len(df) < required:
            return None

        accel_xyz = df[["x", "y", "z"]].to_numpy()  # shape: (n_samples, 3)
        data = Data(array=accel_xyz, sample_rate=FRONTEND_SAMPLING_RATE, channel_names=["x", "y", "z"])
        return data, float(df["t"].min()), float(df["t"].max())

    def get_collection_status(self):
        """Report readiness and progress for the UI."""
        with self.lock:
            n = len(self.buffer)
            required = self.target_total_samples or 0
            if required <= 0:
                return dict(ready=False, samples=n, required=0, duration_s=0.0,
                            required_duration_s=0.0, progress=0.0)

            t_vals = [s["t"] for s in self.buffer] if n else []
            duration = (max(t_vals) - min(t_vals)) if t_vals else 0.0
            progress = min(100.0, (n / required) * 100.0)
            return dict(
                ready=n >= required,
                samples=n,
                required=required,
                duration_s=duration,
                required_duration_s=required / FRONTEND_SAMPLING_RATE,
                progress=progress,
            )

    def get_latest_prediction(self):
        with self.lock:
            return self.predictions[-1] if self.predictions else None

    def score_now(self):
        """Run the pipeline: x,y,z@200Hz → ProcessDynamicAccelerometer → score()."""
        with self.lock:
            win = self._get_window_data()
            if win is None:
                return None
            accel_data, t_min, t_max = win

            try:
                processed = self.processor(accel_data)
                # Let the model decide how many periods based on actual processed duration
                self.model.n_periods = int(
                    np.floor(processed.duration.total_seconds() / self.period_duration)
                )

                preds = score(
                    data=processed,
                    model=self.model,
                    channel_groups=[[0, 1]],   # magnitude + derivative
                    arg_max=False
                )

                conf_values = np.max(preds.array, axis=1)
                class_indices = np.argmax(preds.array, axis=1)

                latest_conf = float(conf_values[-1])
                latest_class = int(class_indices[-1])
                latest_probs = preds.array[-1]

                # Confidence threshold → "Unknown" if low confidence
                if latest_conf < CONFIDENCE_THRESHOLD:
                    label = "Unknown"
                else:
                    label = str(self.labels[latest_class]) if latest_class < len(self.labels) else "Unknown"

                pred = dict(
                    epoch_start=t_min,
                    epoch_end=t_max,
                    label=label,
                    confidence=latest_conf,
                    probabilities={
                        str(self.labels[i]): float(latest_probs[i])
                        for i in range(min(len(self.labels), len(latest_probs)))
                    },
                    timestamp=time.time(),
                    samples_used=len(accel_data.array) if hasattr(accel_data, "array") else 0,
                    n_periods=self.model.n_periods,
                )

                self.predictions.append(pred)
                self.last_prediction_time = t_max

                print(f"[score] {label} (conf={latest_conf:.3f}) "
                      f"| processed_sr≈{getattr(processed,'sample_rate',None)} "
                      f"| chans={getattr(processed,'channel_names',None)} "
                      f"| dur={processed.duration.total_seconds():.2f}s "
                      f"| n_periods={self.model.n_periods}")
                return pred

            except Exception as e:
                print(f"[error] Scoring failed: {e}")
                import traceback
                traceback.print_exc()
                return None


# ========= Boot =========
MODEL_DIR = "./models"
print("[start] Initializing SleepScorer...")
try:
    scorer = SleepScorer(MODEL_DIR)
    print("[start] Model ready.")
except Exception as e:
    print(f"[FATAL] Model load failed: {e}")
    scorer = None

# ========= Routes =========
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": scorer is not None,
        "config": {
            "frontend_sample_rate": FRONTEND_SAMPLING_RATE,
            "model_sample_rate": getattr(scorer, "model_sample_rate", None),
            "period_duration_s": getattr(scorer, "period_duration", None),
            "model_periods": getattr(scorer, "model_periods", None),
            "samples_per_period": getattr(scorer, "model_samples_per_period", None),
            "first_pred_min_periods": getattr(scorer, "min_periods_for_first_pred", None),
            "required_samples": getattr(scorer, "target_total_samples", 0),
            "required_duration_s": (
                scorer.target_total_samples / FRONTEND_SAMPLING_RATE
                if scorer and scorer.target_total_samples else 0
            ),
            "buffer_duration_s": BUFFER_DURATION_S,
            "buffer_size_samples": BUFFER_SIZE,
            "labels": list(getattr(scorer, "labels", ["Wake","NREM","REM"])),
            "confidence_threshold": CONFIDENCE_THRESHOLD,
        },
    })

@app.route("/api/samples", methods=["POST"])
def api_samples():
    if scorer is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json(silent=True) or {}
    samples = data.get("samples", [])
    if not isinstance(samples, list) or not samples:
        return jsonify({"error": "Invalid request"}), 400

    scorer.add_samples(samples)
    status = scorer.get_collection_status()
    pred = scorer.score_now() if status.get("ready") else None

    return jsonify({
        "received": len(samples),
        "buffer_size": len(scorer.buffer),
        "status": status,
        "prediction": pred,
    })

@app.route("/api/latest", methods=["GET"])
def api_latest():
    if scorer is None:
        return jsonify({"error": "Model not loaded"}), 500

    pred = scorer.get_latest_prediction()
    status = scorer.get_collection_status()

    if pred is None:
        msg = "Ready to predict" if status.get("ready") else f"Collecting data... {status.get('progress', 0):.0f}%"
        return jsonify({"status": "no_prediction", "message": msg, "collection_status": status})

    return jsonify({"status": "ok", "prediction": pred, "collection_status": status})

@app.route("/api/reset", methods=["POST"])
def api_reset():
    if scorer is None:
        return jsonify({"error": "Model not loaded"}), 500
    with scorer.lock:
        scorer.buffer.clear()
        scorer.predictions.clear()
        scorer.last_prediction_time = 0.0
    return jsonify({"status": "ok", "message": "Reset complete"})

# ========= Run =========
if __name__ == "__main__":
    if scorer is None:
        print("[FATAL] Cannot start without a model. Ensure ./models/ exists and is valid.")
        raise SystemExit(1)

    print("[start] Flask running on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
