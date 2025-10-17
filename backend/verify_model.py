#!/usr/bin/env python3
"""
Real-time sleep scoring server for Polar H10 accelerometer data.
FIXED: Labels now correctly inferred from actual model output (2 classes).
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
CORS(app)

@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Private-Network", "true")
    return response

# ========= Configuration =========
FRONTEND_SAMPLING_RATE = 200
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))
MIN_PERIODS_FOR_FIRST_PRED = int(os.getenv("MIN_PERIODS_FOR_FIRST_PRED", "2"))
BUFFER_DURATION_S = int(os.getenv("BUFFER_DURATION_S", "300"))
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

        self.model_input_shape = None
        self.model_sample_rate = None
        self.model_periods = None
        self.model_samples_per_period = None
        self.period_duration = None
        self.labels = None
        self.n_output_classes = None

        self.min_periods_for_first_pred = MIN_PERIODS_FOR_FIRST_PRED
        self.target_total_samples = None

        self._load_model(model_dir)

    def _load_model(self, model_dir: str):
        model_path = Path(model_dir).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        print(f"[init] Loading model from: {model_path}")
        self.processor = ProcessDynamicAccelerometer()
        self.model = UTimeModel(model_dir=str(model_path), n_samples_per_prediction=None)

        self.model_input_shape = self.model.input_shape
        self.model_sample_rate = self.model.input_sample_rate
        self.model_periods = self.model_input_shape[0]
        self.model_samples_per_period = self.model_input_shape[1]
        self.period_duration = float(self.model.period_duration)

        print(f"[init] Model loaded from {model_path}")
        print(f"[init] input_shape={self.model_input_shape} | model_sr={self.model_sample_rate} Hz")
        print(f"[init] period_duration={self.period_duration}s | model_periods={self.model_periods}")

        # ============ CRITICAL: Detect actual number of output classes ============
        print(f"[init] *** DETECTING OUTPUT CLASSES ***")
        try:
            test_data = Data(
                array=np.random.randn(64 * 30 * self.model_periods, 2),
                sample_rate=64,
                channel_names=['magnitude', 'magnitude_derivative']
            )
            print(f"[init] Running test prediction with shape: {test_data.array.shape}")
            test_pred = score(
                data=test_data,
                model=self.model,
                channel_groups=[[0, 1]],
                arg_max=False
            )
            print(f"[init] Test prediction output shape: {test_pred.array.shape}")
            self.n_output_classes = test_pred.array.shape[1] if len(test_pred.array.shape) > 1 else 2
            print(f"[init] ✓ DETECTED {self.n_output_classes} OUTPUT CLASSES")
        except Exception as e:
            print(f"[init] ✗ Could not run test prediction: {e}")
            print(f"[init] Defaulting to 2 classes")
            self.n_output_classes = 2

        # ============ Set labels based on actual output classes ============
        if self.n_output_classes == 2:
            self.labels = np.array(["Wake", "Sleep"], dtype=object)
        elif self.n_output_classes == 3:
            self.labels = np.array(["Wake", "NREM", "REM"], dtype=object)
        else:
            self.labels = np.array([f"Class_{i}" for i in range(self.n_output_classes)], dtype=object)

        print(f"[init] *** FINAL LABELS: {list(self.labels)} ***")

        required_seconds = self.min_periods_for_first_pred * self.period_duration
        self.target_total_samples = int(np.ceil(required_seconds * FRONTEND_SAMPLING_RATE))

        print(f"[init] warm-up gate: {self.min_periods_for_first_pred} periods (~{required_seconds:.1f}s) → {self.target_total_samples} samples @ {FRONTEND_SAMPLING_RATE} Hz")
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

        accel_xyz = df[["x", "y", "z"]].to_numpy()
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
        with self.lock:
            win = self._get_window_data()
            if win is None:
                return None
            accel_data, t_min, t_max = win

            try:
                # Process accelerometer (magnitude + derivative)
                processed = self.processor(accel_data)
                
                # Resample to model's expected sample rate (64 Hz)
                if processed.sample_rate != self.model_sample_rate:
                    processed = processed.resample(self.model_sample_rate)
                    print(f"[score] Resampled from {accel_data.sample_rate} Hz to {self.model_sample_rate} Hz")
                
                # Set number of periods
                self.model.n_periods = int(
                    np.floor(processed.duration.total_seconds() / self.period_duration)
                )

                preds = score(
                    data=processed,
                    model=self.model,
                    channel_groups=[[0, 1]],
                    arg_max=False
                )

                conf_values = np.max(preds.array, axis=1)
                class_indices = np.argmax(preds.array, axis=1)

                latest_conf = float(conf_values[-1])
                latest_class = int(class_indices[-1])
                latest_probs = preds.array[-1]

                # Confidence threshold
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

                print(f"[score] {label} (conf={latest_conf:.3f}) | n_periods={self.model.n_periods}")
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
    import traceback
    traceback.print_exc()
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
            "n_output_classes": getattr(scorer, "n_output_classes", None),
            "labels": list(getattr(scorer, "labels", ["Unknown"])),
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

    print("[start] Flask running on http://localhost:5001")
    app.run(host="0.0.0.0", port=5001, debug=True, threaded=True)