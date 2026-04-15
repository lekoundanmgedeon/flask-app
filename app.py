"""
app.py  –  Intel Image Classifier  |  Flask Backend
Supports PyTorch (.pth) and TensorFlow (.keras) models.
GPU is used automatically when available.
"""

import os
import io
import sys
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template, abort

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB upload limit

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES  = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
CLASS_EMOJIS = ["🏙️",        "🌲",     "🧊",      "⛰️",       "🌊",  "🛣️"]
IMG_SIZE     = 150
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Model paths – resolved relative to this file
BASE_DIR    = Path(__file__).parent
MODELS_DIR  = BASE_DIR / "models"
PT_MODEL    = MODELS_DIR/ "intel_model.pth"
TF_MODEL    = MODELS_DIR / "intel_model.h5"

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ── Lazy model cache ──────────────────────────────────────────────────────────
_models: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# Model loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_pytorch_model():
    """Load the PyTorch CNN; returns (model, device)."""
    if "pytorch" in _models:
        return _models["pytorch"]

    import torch
    # Import architecture from the training module
    sys.path.insert(0, str(BASE_DIR))
    from models.pytorch_model import IntelCNN

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    log.info("[PyTorch] Using device: %s", device)

    ckpt  = torch.load(str(PT_MODEL), map_location=device, weights_only=False)
    model = IntelCNN(num_classes=len(CLASS_NAMES))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    _models["pytorch"] = (model, device)
    log.info("[PyTorch] Model loaded from %s", PT_MODEL)
    return _models["pytorch"]


# In app.py, replace load_tensorflow_model() with this:

def load_tensorflow_model():
    if "tensorflow" in _models:
        return _models["tensorflow"]

    import tensorflow as tf
    from tensorflow.keras.layers import Dense

    # Keras 2 doesn't know 'quantization_config' — strip it before init
    class PatchedDense(Dense):
        def __init__(self, *args, **kwargs):
            kwargs.pop("quantization_config", None)
            super().__init__(*args, **kwargs)

    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    log.info("[TensorFlow] GPUs detected: %d", len(gpus))

    model = tf.keras.models.load_model(
        str(TF_MODEL),
        compile=False,
        custom_objects={"Dense": PatchedDense}
    )
    _models["tensorflow"] = model
    log.info("[TensorFlow] Model loaded from %s", TF_MODEL)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(img_bytes: bytes) -> np.ndarray:
    """PIL → float32 array normalised with ImageNet stats, shape (H,W,3)."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def predict_pytorch(img_bytes: bytes) -> list[float]:
    import torch
    model, device = load_pytorch_model()
    arr    = preprocess(img_bytes)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze().cpu().numpy()
    return probs.tolist()


def predict_tensorflow(img_bytes: bytes) -> list[float]:
    model = load_tensorflow_model()
    arr   = preprocess(img_bytes)
    batch = np.expand_dims(arr, 0)          # (1, H, W, 3)
    probs = model.predict(batch, verbose=0)[0]
    return probs.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html",
                           class_names=CLASS_NAMES,
                           class_emojis=CLASS_EMOJIS)


@app.route("/predict", methods=["POST"])
def predict():
    # ── Validate inputs ───────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file  = request.files["image"]
    model_choice = request.form.get("model", "pytorch").lower()

    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        return jsonify({"error": f"Unsupported file type '{ext}'."}), 400

    if model_choice not in ("pytorch", "tensorflow"):
        return jsonify({"error": "model must be 'pytorch' or 'tensorflow'."}), 400

    # ── Run inference ─────────────────────────────────────────
    try:
        img_bytes = file.read()

        if model_choice == "pytorch":
            probs = predict_pytorch(img_bytes)
        else:
            probs = predict_tensorflow(img_bytes)

        top_idx  = int(np.argmax(probs))
        result   = {
            "predicted_class": CLASS_NAMES[top_idx],
            "emoji":           CLASS_EMOJIS[top_idx],
            "confidence":      round(probs[top_idx] * 100, 2),
            "probabilities":   {
                CLASS_NAMES[i]: round(probs[i] * 100, 2)
                for i in range(len(CLASS_NAMES))
            },
            "model_used": model_choice,
        }
        log.info("Prediction: %s (%.1f%%) via %s",
                 result["predicted_class"], result["confidence"], model_choice)
        return jsonify(result)

    except Exception as exc:
        log.exception("Inference failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "loaded_models": list(_models.keys())})


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Pre-warm models on startup
    log.info("Pre-loading models…")
    try:
        load_pytorch_model()
        log.info("PyTorch model ready.")
    except Exception as e:
        log.warning("Could not pre-load PyTorch model: %s", e)
    try:
        load_tensorflow_model()
        log.info("TensorFlow model ready.")
    except Exception as e:
        log.warning("Could not pre-load TensorFlow model: %s", e)

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
