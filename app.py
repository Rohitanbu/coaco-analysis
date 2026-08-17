#!/usr/bin/env python3
"""Cocoa ripeness analyser API + template frontend.

Dual-upload ``POST /api/predict`` (thermal JPG + acoustic TDMS/CSV) returns
prediction probabilities, extracted features, downsampled waveform, and a
playable WAV URL. Legacy single-file upload is still supported for older clients.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import uuid
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import models, transforms

from utils.audio_generator import write_wav
from utils.db import init_db, save_run
from utils.signal_processing import (
    downsample_waveform,
    extract_features,
    parse_acoustic_file,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = ROOT / "templatemo_614_quantix_saas"
STATIC_DIR = WEB_DIR / "static"
AUDIO_DIR = STATIC_DIR / "audio"
CLASS_CKPT = ROOT / "thermal_classifier" / "outputs" / "best_efficientnet_b0.pt"
ACOUSTIC_CKPT = ROOT / "acoustic_classifier" / "outputs" / "best_efficientnet_b0_acoustic.pt"
METRICS_CKPT = WEB_DIR / "outputs" / "metrics_regressor.joblib"
PROFILES_PATH = WEB_DIR / "outputs" / "class_metric_profiles.json"

CLASSES = ["OR", "R", "UR"]
CLASS_LABELS = {
    "OR": "Overripe",
    "R": "Ripe",
    "UR": "Unripe",
}

# SWAP-IN POINT: replace fuse_cnn_probabilities() with an SVM predict_proba call
# once a trained fused model artifact is available (e.g. joblib.load("svm_fused.joblib")).
THERMAL_WEIGHT = 0.55
ACOUSTIC_WEIGHT = 0.45

app = FastAPI(title="Cocoa Ripeness Analyser", version="2.0.0")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_state: dict = {"ready": False}


def build_classifier(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def load_models() -> None:
    if not CLASS_CKPT.exists():
        raise FileNotFoundError(f"Missing classifier checkpoint: {CLASS_CKPT}")
    if not PROFILES_PATH.exists():
        raise FileNotFoundError(f"Missing class profiles: {PROFILES_PATH}")

    ckpt = torch.load(CLASS_CKPT, map_location=device, weights_only=False)
    classes = ckpt.get("classes", CLASSES)
    clf = build_classifier(len(classes))
    clf.load_state_dict(ckpt["model_state"])
    clf.to(device).eval()

    acoustic_clf = None
    acoustic_classes: list[str] = []
    if ACOUSTIC_CKPT.exists():
        ackpt = torch.load(ACOUSTIC_CKPT, map_location=device, weights_only=False)
        acoustic_classes = ackpt.get("classes", ["R", "UR"])
        acoustic_clf = build_classifier(len(acoustic_classes))
        acoustic_clf.load_state_dict(ackpt["model_state"])
        acoustic_clf.to(device).eval()

    profiles = json.loads(PROFILES_PATH.read_text())

    regressor = None
    backbone = None
    target_names = ["amplitude", "frequency", "power"]
    log1p = True
    if METRICS_CKPT.exists():
        bundle = joblib.load(METRICS_CKPT)
        regressor = bundle["regressor"]
        target_names = bundle.get("target_names", target_names)
        log1p = bundle.get("log1p", True)
        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        backbone.classifier = nn.Identity()
        backbone.to(device).eval()

    _state.update(
        {
            "ready": True,
            "classes": classes,
            "classifier": clf,
            "acoustic_classes": acoustic_classes,
            "acoustic_classifier": acoustic_clf,
            "profiles": profiles,
            "backbone": backbone,
            "regressor": regressor,
            "target_names": target_names,
            "log1p": log1p,
        }
    )


TF = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

FS = 12800
N_SAMPLES = int(FS * 4.0)


def read_image(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc


def hz_to_mel(f: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(n_fft: int, n_mels: int, fs: int) -> np.ndarray:
    f_max = fs / 2.0
    mels = np.linspace(hz_to_mel(np.array([0.0]))[0], hz_to_mel(np.array([f_max]))[0], n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / fs).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center == left:
            center += 1
        if right == center:
            right += 1
        for j in range(left, center):
            if 0 <= j < fb.shape[1]:
                fb[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if 0 <= j < fb.shape[1]:
                fb[i, j] = (right - j) / (right - center)
    return fb


def mel_spectrogram(
    x: np.ndarray, fs: int, n_fft: int = 1024, hop: int = 256, n_mels: int = 128
) -> np.ndarray:
    window = np.hanning(n_fft).astype(np.float32)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n_frames = 1 + (len(x) - n_fft) // hop
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n_frames, n_fft), strides=(x.strides[0] * hop, x.strides[0]), writeable=False
    ).copy()
    frames *= window
    spec = np.fft.rfft(frames, axis=1)
    power = (spec.real**2 + spec.imag**2).astype(np.float32)
    fb = mel_filterbank(n_fft, n_mels, fs)
    mel = fb @ power.T
    mel = np.log1p(mel)
    mel = mel - mel.min()
    return (mel / (mel.max() + 1e-8)).astype(np.float32)


def _prepare_acoustic_window(voltage: np.ndarray, fs: int) -> np.ndarray:
    x = voltage.astype(np.float64)
    x = x - np.mean(x)
    target = int(fs * 4.0)
    if len(x) >= target:
        start = (len(x) - target) // 2
        x = x[start : start + target]
    else:
        pad = target - len(x)
        x = np.pad(x, (pad // 2, pad - pad // 2))
    return x.astype(np.float32)


def fuse_cnn_probabilities(
    thermal_probs: dict[str, float],
    acoustic_probs: dict[str, float],
) -> list[dict[str, float | str]]:
    """Fuse thermal + acoustic CNN softmax outputs (placeholder for fused SVM)."""
    fused: dict[str, float] = {}
    for code in CLASSES:
        tp = thermal_probs.get(code, 0.0)
        ap = acoustic_probs.get(code, 0.0)
        fused[code] = THERMAL_WEIGHT * tp + ACOUSTIC_WEIGHT * ap

    total = sum(fused.values()) or 1.0
    fused = {k: v / total for k, v in fused.items()}
    ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
    return [
        {"label": CLASS_LABELS[code], "confidence": round(prob, 4)}
        for code, prob in ranked
    ]


@torch.no_grad()
def thermal_probabilities(img: Image.Image) -> dict[str, float]:
    if not _state["ready"]:
        load_models()
    tensor = TF(img).unsqueeze(0).to(device)
    logits = _state["classifier"](tensor)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    classes = _state["classes"]
    return {c: float(probs[i]) for i, c in enumerate(classes)}


@torch.no_grad()
def acoustic_probabilities(voltage: np.ndarray, fs: int) -> dict[str, float]:
    if not _state["ready"]:
        load_models()
    if _state["acoustic_classifier"] is None:
        raise RuntimeError("Acoustic classifier model not found")

    x = _prepare_acoustic_window(voltage, fs)
    mel = mel_spectrogram(x, fs)
    img = torch.from_numpy(mel).unsqueeze(0)
    img = transforms.Resize((224, 224), antialias=True)(img).repeat(3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    tensor = ((img - mean) / std).unsqueeze(0).float().to(device)

    logits = _state["acoustic_classifier"](tensor)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    classes = _state["acoustic_classes"]
    return {c: float(probs[i]) for i, c in enumerate(classes)}


def build_api_response(
    *,
    run_id: str,
    probabilities: list[dict],
    features: dict,
    waveform_samples: list[float],
    sample_rate_hz: float,
) -> dict:
    top = probabilities[0]
    return {
        "run_id": run_id,
        "prediction": {
            "label": top["label"],
            "confidence": top["confidence"],
        },
        "probabilities": probabilities,
        "features": features,
        "waveform": {
            "sample_rate_hz": round(sample_rate_hz, 2),
            "samples": waveform_samples,
        },
        "audio_url": f"/api/audio/{run_id}.wav",
    }


async def predict_fused(thermal_image: UploadFile, acoustic_file: UploadFile) -> dict:
    if not _state["ready"]:
        load_models()

    thermal_data = await thermal_image.read()
    acoustic_data = await acoustic_file.read()
    if not thermal_data or not acoustic_data:
        raise HTTPException(status_code=400, detail="Both uploads must be non-empty")

    img = read_image(thermal_data)
    try:
        signal = parse_acoustic_file(acoustic_data, acoustic_file.filename or "audio.csv")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    feats = extract_features(signal.voltage)
    features = {
        "amplitude": round(feats.amplitude, 6),
        "rms": round(feats.rms, 6),
        "power": round(feats.power, 6),
        "frequency": None,  # out of scope — no fabricated frequency metric
    }

    try:
        t_probs = thermal_probabilities(img)
        a_probs = acoustic_probabilities(signal.voltage, signal.sample_rate_hz)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    probabilities = fuse_cnn_probabilities(t_probs, a_probs)
    run_id = str(uuid.uuid4())
    wav_path = AUDIO_DIR / f"{run_id}.wav"
    write_wav(signal.voltage, signal.sample_rate_hz, wav_path)

    save_run(
        run_id=run_id,
        predicted_label=probabilities[0]["label"],
        confidence=float(probabilities[0]["confidence"]),
        thermal_filename=thermal_image.filename,
        acoustic_filename=acoustic_file.filename,
        audio_path=str(wav_path),
        features=features,
        probabilities=probabilities,
    )

    return build_api_response(
        run_id=run_id,
        probabilities=probabilities,
        features=features,
        waveform_samples=downsample_waveform(signal.voltage),
        sample_rate_hz=signal.sample_rate_hz,
    )


# --- Legacy single-file helpers (webapp/static) ---

def load_signal_from_bytes(data: bytes) -> np.ndarray:
    vals: list[float] = []
    reader = csv.reader(io.StringIO(data.decode("utf-8-sig")))
    next(reader, None)
    for row in reader:
        if len(row) >= 2:
            try:
                vals.append(float(row[1]))
            except ValueError:
                continue
    x = np.asarray(vals, dtype=np.float64)
    if len(x) == 0:
        raise ValueError("Empty signal")
    x = x - np.mean(x)
    if len(x) >= N_SAMPLES:
        start = (len(x) - N_SAMPLES) // 2
        x = x[start : start + N_SAMPLES]
    else:
        pad = N_SAMPLES - len(x)
        x = np.pad(x, (pad // 2, pad - pad // 2))
    return x.astype(np.float32)


@torch.no_grad()
def predict_acoustic_legacy(csv_bytes: bytes) -> dict:
    x = load_signal_from_bytes(csv_bytes)
    probs = acoustic_probabilities(x, FS)
    idx_label = max(probs, key=probs.get)
    feats = extract_features(x)
    return {
        "quality": idx_label,
        "quality_label": CLASS_LABELS.get(idx_label, idx_label),
        "confidence": probs[idx_label],
        "class_probabilities": probs,
        "amplitude": {"value": feats.amplitude, "unit": "peak (V)"},
        "frequency": {"value": None, "unit": "Hz"},
        "power": {"value": feats.power, "unit": "mean-square"},
        "metrics_source": "extracted from signal",
    }


@torch.no_grad()
def predict_thermal_legacy(img: Image.Image) -> dict:
    probs = thermal_probabilities(img)
    idx_label = max(probs, key=probs.get)
    profile = _state["profiles"][idx_label]
    return {
        "quality": idx_label,
        "quality_label": CLASS_LABELS.get(idx_label, idx_label),
        "confidence": probs[idx_label],
        "class_probabilities": probs,
        "amplitude": profile["amplitude"],
        "frequency": profile["frequency"],
        "power": profile["power"],
        "metrics_source": "class_profile",
    }


@app.on_event("startup")
def startup() -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    init_db()
    try:
        load_models()
        logger.info("Models loaded on %s", device)
    except Exception as exc:
        logger.warning("Startup warning: %s", exc)


@app.get("/api/health")
def health():
    return {
        "ok": _state.get("ready", False),
        "device": str(device),
        "classifier": CLASS_CKPT.exists(),
        "acoustic_classifier": ACOUSTIC_CKPT.exists(),
        "profiles": PROFILES_PATH.exists(),
    }


@app.get("/api/audio/{run_id}.wav")
def get_audio(run_id: str):
    safe_id = run_id.removesuffix(".wav")
    path = AUDIO_DIR / f"{safe_id}.wav"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(path, media_type="audio/wav", filename=f"{safe_id}.wav")


@app.post("/api/predict")
async def predict(
    thermal_image: UploadFile | None = File(None),
    acoustic_file: UploadFile | None = File(None),
    file: UploadFile | None = File(None),
):
    """Dual upload (thermal + acoustic) returns Step-1 contract; legacy single ``file`` still supported."""
    if thermal_image is not None and acoustic_file is not None:
        return await predict_fused(thermal_image, acoustic_file)

    if file is None:
        raise HTTPException(
            status_code=400,
            detail="Provide thermal_image and acoustic_file, or a single legacy file upload",
        )

    if not _state["ready"]:
        load_models()

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    is_csv = (file.filename or "").lower().endswith(".csv")
    is_image = (file.content_type or "").startswith("image/")

    try:
        if is_csv:
            result = predict_acoustic_legacy(data)
        elif is_image:
            result = predict_thermal_legacy(read_image(data))
        else:
            raise HTTPException(status_code=400, detail="Upload an image or CSV file")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    result["filename"] = file.filename
    return result


@app.post("/api/predict/batch")
async def predict_batch(request: Request):
    """Batch upload (multiple thermal + acoustic pairs)."""
    form = await request.form()
    samples = {}
    
    for key, val in form.multi_items():
        if key.startswith("sample_"):
            parts = key.split("_", 2)
            if len(parts) == 3:
                idx = parts[1]
                field = parts[2] # 'thermal_image' or 'acoustic_file'
                if idx not in samples:
                    samples[idx] = {}
                samples[idx][field] = val
                
    if not samples:
        raise HTTPException(status_code=400, detail="No samples provided in batch")
        
    results = []
    # Sort by sample index to maintain order
    for idx, data in sorted(samples.items(), key=lambda x: int(x[0])):
        t_img = data.get("thermal_image")
        a_file = data.get("acoustic_file")
        if t_img and a_file:
            res = await predict_fused(t_img, a_file)
            res["sample_id"] = idx
            results.append(res)
            
    return {"results": results}



# Template frontend (register API routes above before this mount)
app.mount("/", StaticFiles(directory=str(TEMPLATE_DIR), html=True), name="frontend")
