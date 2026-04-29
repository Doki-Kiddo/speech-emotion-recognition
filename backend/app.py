import json
import os
import sys
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LABELS_PATH, MODEL_PATH

FRONTEND_DIR = PROJECT_ROOT / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)

model = None
labels = None
model_mtime = None
labels_mtime = None


def load_assets():
    global model, labels, model_mtime, labels_mtime
    model_file = PROJECT_ROOT / MODEL_PATH
    labels_file = PROJECT_ROOT / LABELS_PATH
    if not model_file.exists() or not labels_file.exists():
        raise FileNotFoundError("Train the model first with: python train.py --data-dir data/raw")

    current_model_mtime = model_file.stat().st_mtime
    current_labels_mtime = labels_file.stat().st_mtime
    needs_reload = (
        model is None
        or labels is None
        or model_mtime != current_model_mtime
        or labels_mtime != current_labels_mtime
    )

    if needs_reload:
        print("Loading TensorFlow model...", flush=True)
        from tensorflow.keras.models import load_model

        model = load_model(model_file)
        labels = json.loads(labels_file.read_text(encoding="utf-8"))
        model_mtime = current_model_mtime
        labels_mtime = current_labels_mtime
        print("Model loaded.", flush=True)
    return model, labels


def warmup():
    load_assets()
    print("Loading audio feature tools...", flush=True)
    import numpy
    import src.features

    print("Backend warmup complete.", flush=True)


@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/api/health")
def health():
    model_file = PROJECT_ROOT / MODEL_PATH
    labels_file = PROJECT_ROOT / LABELS_PATH
    trained = model_file.exists() and labels_file.exists()
    return jsonify(
        {
            "ok": True,
            "trained": trained,
            "model_mtime": model_file.stat().st_mtime if model_file.exists() else None,
        }
    )


@app.post("/api/predict")
def predict():
    try:
        if "audio" not in request.files:
            return jsonify({"error": "Upload field must be named audio"}), 400

        print("Prediction request received.", flush=True)
        import numpy as np
        from src.features import audio_to_mel, load_audio_bytes

        clf, class_names = load_assets()
        upload = request.files["audio"]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            upload.save(tmp_path)
            print("Audio saved, extracting features...", flush=True)
            audio = load_audio_bytes(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        feature = audio_to_mel(audio)
        print("Running model prediction...", flush=True)
        probabilities = clf.predict(np.expand_dims(feature, axis=0), verbose=0)[0]
        scores = {label: float(prob) for label, prob in zip(class_names, probabilities)}
        emotion = max(scores, key=scores.get)
        print(f"Prediction complete: {emotion}", flush=True)
        return jsonify({"emotion": emotion, "scores": scores})
    except Exception as exc:
        print(f"Prediction failed: {exc}", flush=True)
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    print("Starting app at http://127.0.0.1:5000", flush=True)
    try:
        warmup()
    except Exception as exc:
        print(f"Model warmup skipped: {exc}", flush=True)
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False, threaded=True)
