#!/usr/bin/env python3
"""4-class cocoa thermal image classifier (EfficientNet-B0).

Pod-level stratified 80/20 split so Front/Back views of the same pod
never leak across train and test.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

CLASSES = ["CPB", "OR", "R", "UR"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
POD_RE = re.compile(r"^(CPB|OR|R|UR)(\d+)([FBfb])\.(jpg|jpeg|png)$", re.I)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def discover_pods(data_root: Path) -> dict[str, list[dict]]:
    """Map class -> list of {pod_id, paths}."""
    by_class: dict[str, dict[str, list[Path]]] = {c: defaultdict(list) for c in CLASSES}
    for cls in CLASSES:
        folder = data_root / cls
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing class folder: {folder}")
        for path in sorted(folder.iterdir()):
            if not path.is_file():
                continue
            m = POD_RE.match(path.name)
            if not m:
                continue
            label, pod_num, _side, _ext = m.groups()
            label = label.upper()
            if label != cls:
                continue
            by_class[cls][f"{cls}{pod_num}"].append(path)

    pods: dict[str, list[dict]] = {}
    for cls in CLASSES:
        pods[cls] = [
            {"pod_id": pod_id, "paths": paths}
            for pod_id, paths in sorted(by_class[cls].items())
        ]
        if not pods[cls]:
            raise RuntimeError(f"No images found for class {cls}")
    return pods


def pod_level_split(
    pods: dict[str, list[dict]], test_size: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Stratified split on pod IDs; expand to image records."""
    train_records: list[dict] = []
    test_records: list[dict] = []

    for cls, pod_list in pods.items():
        labels = [cls] * len(pod_list)
        train_pods, test_pods = train_test_split(
            pod_list,
            test_size=test_size,
            random_state=seed,
            stratify=labels if len(pod_list) >= 2 else None,
        )
        for split, bucket in ((train_pods, train_records), (test_pods, test_records)):
            for pod in split:
                for path in pod["paths"]:
                    bucket.append(
                        {
                            "path": path,
                            "label": CLASS_TO_IDX[cls],
                            "class": cls,
                            "pod_id": pod["pod_id"],
                        }
                    )
    return train_records, test_records


class ThermalDataset(Dataset):
    def __init__(self, records: list[dict], transform):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        img = Image.open(rec["path"]).convert("RGB")
        return self.transform(img), rec["label"]


def build_model(num_classes: int, device: torch.device) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model.to(device)


def train_one_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    return (
        total_loss / max(total, 1),
        correct / max(total, 1),
        np.asarray(y_true),
        np.asarray(y_pred),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "Acoustics_Extracted"
        / "Thermal Images",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "auto"],
        help="Force cuda/cpu, or auto-detect",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but torch.cuda.is_available() is False. "
                "Install a CUDA build of PyTorch."
            )
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"CUDA: {torch.version.cuda}", flush=True)
    print(f"Data root: {args.data_root}", flush=True)

    pods = discover_pods(args.data_root)
    train_records, test_records = pod_level_split(pods, args.test_size, args.seed)

    train_pods = sorted({r["pod_id"] for r in train_records})
    test_pods = sorted({r["pod_id"] for r in test_records})
    assert not (set(train_pods) & set(test_pods)), "Pod leakage between splits"

    print("Pod counts by class:")
    for cls in CLASSES:
        n_train = sum(1 for p in train_pods if p.startswith(cls))
        n_test = sum(1 for p in test_pods if p.startswith(cls))
        print(f"  {cls}: train_pods={n_train} test_pods={n_test}")
    print(
        f"Images: train={len(train_records)} test={len(test_records)} "
        f"(train_pods={len(train_pods)} test_pods={len(test_pods)})"
    )
    print("Train class counts:", Counter(r["class"] for r in train_records))
    print("Test class counts:", Counter(r["class"] for r in test_records))

    train_tf = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    train_ds = ThermalDataset(train_records, train_tf)
    test_ds = ThermalDataset(test_records, eval_tf)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(len(CLASSES), device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best_acc = -1.0
    best_path = args.out_dir / "best_efficientnet_b0.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f}  "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.3f}"
        )
        if test_acc >= best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": CLASSES,
                    "epoch": epoch,
                    "test_acc": test_acc,
                    "args": vars(args),
                },
                best_path,
            )

    # Final eval with best checkpoint
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    _, best_test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    report = classification_report(
        y_true, y_pred, target_names=CLASSES, digits=4, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred).tolist()
    report_text = classification_report(
        y_true, y_pred, target_names=CLASSES, digits=4
    )

    metrics = {
        "device": str(device),
        "best_epoch": ckpt["epoch"],
        "best_test_acc": best_test_acc,
        "train_images": len(train_records),
        "test_images": len(test_records),
        "train_pods": len(train_pods),
        "test_pods": len(test_pods),
        "classes": CLASSES,
        "classification_report": report,
        "confusion_matrix": cm,
        "history": history,
        "split": {
            "strategy": "pod-level stratified 80/20",
            "test_size": args.test_size,
            "seed": args.seed,
            "train_pods": train_pods,
            "test_pods": test_pods,
        },
    }
    metrics_path = args.out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
    (args.out_dir / "classification_report.txt").write_text(report_text)

    print("\n=== Best model ===")
    print(f"Checkpoint: {best_path}")
    print(f"Best epoch: {ckpt['epoch']}  test_acc={best_test_acc:.4f}")
    print(report_text)
    print("Confusion matrix (rows=true, cols=pred):")
    print(np.array(cm))
    print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
