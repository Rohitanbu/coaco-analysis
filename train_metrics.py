#!/usr/bin/env python3
"""Train thermal-image -> acoustic metric regressor.

Targets (from paired acoustic CSVs of the same pod):
  - amplitude: RMS of centered signal
  - frequency: spectral centroid (Hz)
  - power: mean square of centered signal
"""

from __future__ import annotations

import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
THERMAL_ROOT = ROOT / "Acoustics_Extracted" / "Thermal Images"
ACOUSTIC_ROOT = ROOT / "Acoustics_Extracted" / "Acoustics"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
CLASSES = ["CPB", "OR", "R", "UR"]
POD_IMG_RE = re.compile(r"^(CPB|OR|R|UR)(\d+)([FBfb])\.(jpg|jpeg|png)$", re.I)
POD_CSV_RE = re.compile(r"^(CPB|OR|R|UR)(\d+)", re.I)
FS = 12800.0
N_SAMPLES = int(FS * 4)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_signal(path: Path) -> np.ndarray:
    vals = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                try:
                    vals.append(float(row[1]))
                except ValueError:
                    continue
    x = np.asarray(vals, dtype=np.float64)
    x = x - x.mean()
    if len(x) >= N_SAMPLES:
        start = (len(x) - N_SAMPLES) // 2
        x = x[start : start + N_SAMPLES]
    else:
        pad = N_SAMPLES - len(x)
        x = np.pad(x, (pad // 2, pad - pad // 2))
    return x


def acoustic_metrics(x: np.ndarray) -> dict[str, float]:
    amp = float(np.sqrt(np.mean(x**2)))  # RMS amplitude
    power = float(np.mean(x**2))
    # spectral centroid
    n_fft = 2048
    window = np.hanning(n_fft)
    hop = 512
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n_frames = 1 + (len(x) - n_fft) // hop
    cents = []
    freqs = np.fft.rfftfreq(n_fft, 1.0 / FS)
    for i in range(n_frames):
        frame = x[i * hop : i * hop + n_fft] * window
        ps = np.abs(np.fft.rfft(frame)) ** 2
        cents.append(float((freqs * ps).sum() / (ps.sum() + 1e-12)))
    freq = float(np.mean(cents)) if cents else 0.0
    return {"amplitude": amp, "frequency": freq, "power": power}


def discover_pairs() -> list[dict]:
    """Pair thermal images with mean acoustic metrics of the same pod."""
    # acoustic metrics by pod
    metrics_by_pod: dict[str, list[dict]] = defaultdict(list)
    for cls in CLASSES:
        folder = ACOUSTIC_ROOT / cls
        if not folder.is_dir():
            continue
        for path in folder.glob("*.csv"):
            m = POD_CSV_RE.match(path.name)
            if not m:
                continue
            pod_id = f"{m.group(1).upper()}{m.group(2)}"
            try:
                metrics_by_pod[pod_id].append(acoustic_metrics(load_signal(path)))
            except Exception:
                continue

    pod_targets = {}
    for pod_id, rows in metrics_by_pod.items():
        pod_targets[pod_id] = {
            "amplitude": float(np.mean([r["amplitude"] for r in rows])),
            "frequency": float(np.mean([r["frequency"] for r in rows])),
            "power": float(np.mean([r["power"] for r in rows])),
            "class": re.match(r"^(CPB|OR|R|UR)", pod_id).group(1),
        }

    pairs = []
    for cls in CLASSES:
        folder = THERMAL_ROOT / cls
        if not folder.is_dir():
            continue
        for path in folder.iterdir():
            m = POD_IMG_RE.match(path.name)
            if not m:
                continue
            pod_id = f"{m.group(1).upper()}{m.group(2)}"
            if pod_id not in pod_targets:
                continue
            t = pod_targets[pod_id]
            pairs.append(
                {
                    "path": path,
                    "pod_id": pod_id,
                    "class": t["class"],
                    "amplitude": t["amplitude"],
                    "frequency": t["frequency"],
                    "power": t["power"],
                }
            )
    return pairs


class EmbedDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(rec["path"]).convert("RGB")
        y = np.array(
            [rec["amplitude"], rec["frequency"], rec["power"]], dtype=np.float32
        )
        return self.transform(img), y, rec["class"]


@torch.no_grad()
def extract_embeddings(records, device):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    backbone = models.efficientnet_b0(weights=weights)
    backbone.classifier = nn.Identity()
    backbone = backbone.to(device).eval()
    tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    ds = EmbedDataset(records, tf)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)
    embs, ys, labels = [], [], []
    for imgs, y, cls in tqdm(loader, desc="embeddings"):
        feats = backbone(imgs.to(device)).cpu().numpy()
        embs.append(feats)
        ys.append(y.numpy())
        labels.extend(cls)
    return np.vstack(embs), np.vstack(ys), labels


def main():
    set_seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pairs = discover_pairs()
    print(f"Paired thermal images with acoustic metrics: {len(pairs)}")
    if len(pairs) < 20:
        raise RuntimeError("Not enough paired samples to train metrics model")

    # pod-level split
    pods = sorted({p["pod_id"] for p in pairs})
    pod_cls = {p["pod_id"]: p["class"] for p in pairs}
    y_pod = [pod_cls[p] for p in pods]
    # OR may have only 1 paired pod — fall back if stratify is impossible
    counts = {c: y_pod.count(c) for c in set(y_pod)}
    can_stratify = all(v >= 2 for v in counts.values())
    train_pods, test_pods = train_test_split(
        pods,
        test_size=0.2,
        random_state=42,
        stratify=y_pod if can_stratify else None,
    )
    train_pods, test_pods = set(train_pods), set(test_pods)
    train_recs = [p for p in pairs if p["pod_id"] in train_pods]
    test_recs = [p for p in pairs if p["pod_id"] in test_pods]
    print(f"Train images={len(train_recs)} Test images={len(test_recs)}")

    X_train, y_train, _ = extract_embeddings(train_recs, device)
    X_test, y_test, _ = extract_embeddings(test_recs, device)

    # log-transform targets for stability (especially power/amplitude)
    y_train_log = np.log1p(np.maximum(y_train, 0))
    y_test_log = np.log1p(np.maximum(y_test, 0))

    model = MultiOutputRegressor(
        GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
        )
    )
    model.fit(X_train, y_train_log)
    pred_log = model.predict(X_test)
    pred = np.expm1(pred_log)

    names = ["amplitude", "frequency", "power"]
    metrics = {}
    for i, name in enumerate(names):
        err = pred[:, i] - y_test[:, i]
        mae = float(np.mean(np.abs(err)))
        mape = float(np.mean(np.abs(err) / (np.abs(y_test[:, i]) + 1e-8)) * 100)
        # R^2
        ss_res = float(np.sum(err**2))
        ss_tot = float(np.sum((y_test[:, i] - y_test[:, i].mean()) ** 2)) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        metrics[name] = {"mae": mae, "mape_pct": mape, "r2": r2}
        print(f"{name}: MAE={mae:.6f}  MAPE={mape:.1f}%  R2={r2:.3f}")

    joblib.dump(
        {
            "regressor": model,
            "target_names": names,
            "log1p": True,
            "units": {
                "amplitude": "RMS (a.u.)",
                "frequency": "Hz",
                "power": "mean-square (a.u.)",
            },
        },
        OUT_DIR / "metrics_regressor.joblib",
    )
    (OUT_DIR / "metrics_regressor_report.json").write_text(
        json.dumps(
            {
                "n_train": len(train_recs),
                "n_test": len(test_recs),
                "targets": metrics,
            },
            indent=2,
        )
    )
    print(f"Saved {OUT_DIR / 'metrics_regressor.joblib'}")


if __name__ == "__main__":
    main()
