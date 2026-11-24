"""Training pipeline for AcousticMobileNet with CUDA Conv2d/Linear smoke check and CPU fallback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import (
    CLASSES,
    CLASS_TO_IDX,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_METRICS_PATH,
    FMAX,
    FMIN,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    NUM_FRAMES,
    NUM_SAMPLES,
    SAMPLE_RATE,
)
from feature_extraction import extract_batch_spectrograms
from model import AcousticMobileNet
from synthetic_dataset import generate_synthetic_dataset


def resolve_training_device(prefer_cuda: bool = True) -> tuple[torch.device, str]:
    """Select CUDA only after passing a real Conv2d + Linear kernel smoke test."""
    if not prefer_cuda or not torch.cuda.is_available():
        return torch.device("cpu"), "CUDA unavailable; using CPU."

    try:
        conv = nn.Conv2d(1, 8, kernel_size=3, padding=1).cuda()
        pool = nn.AdaptiveAvgPool2d(1).cuda()
        fc = nn.Linear(8, len(CLASSES)).cuda()
        x = torch.randn(2, 1, 32, 32, device="cuda", dtype=torch.float32)
        out = fc(pool(conv(x)).flatten(1))
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()
        return torch.device("cuda"), f"CUDA verified on {torch.cuda.get_device_name(0)}"
    except Exception as exc:
        return torch.device("cpu"), f"CUDA smoke test failed ({exc}); falling back to CPU."


def _load_wav_directory_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load 1-second mono 16kHz WAVs arranged under data_dir/<class_name>/*.wav."""
    clips: list[np.ndarray] = []
    labels: list[int] = []

    for class_name in CLASSES:
        cls_dir = data_dir / class_name
        if not cls_dir.exists():
            continue
        class_idx = CLASS_TO_IDX[class_name]
        for wav_path in sorted(cls_dir.rglob("*.wav")):
            audio, orig_sr = sf.read(wav_path, dtype="float32")
            arr = np.asarray(audio, dtype=np.float32)
            if arr.ndim > 1:
                arr = np.mean(arr, axis=-1, dtype=np.float32)
            if int(orig_sr) != SAMPLE_RATE and arr.size > 0:
                from scipy import signal

                gcd = int(np.gcd(int(orig_sr), SAMPLE_RATE))
                arr = signal.resample_poly(arr, SAMPLE_RATE // gcd, int(orig_sr) // gcd).astype(np.float32)
            if arr.size < NUM_SAMPLES:
                arr = np.pad(arr, (0, NUM_SAMPLES - arr.size))
                clips.append(arr)
                labels.append(class_idx)
            else:
                for start in range(0, arr.size - NUM_SAMPLES + 1, NUM_SAMPLES):
                    clips.append(arr[start : start + NUM_SAMPLES])
                    labels.append(class_idx)

    if not clips:
        raise FileNotFoundError(f"No WAV samples found in class subdirectories under {data_dir}.")

    return np.stack(clips, axis=0).astype(np.float32), np.asarray(labels, dtype=np.int64)


def _stratified_split(
    features: np.ndarray,
    labels: np.ndarray,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []

    for cls_idx in range(len(CLASSES)):
        cls_indices = np.where(labels == cls_idx)[0]
        if cls_indices.size == 0:
            continue
        rng.shuffle(cls_indices)
        n_val = max(1, int(round(cls_indices.size * val_fraction)))
        if n_val >= cls_indices.size:
            n_val = max(1, cls_indices.size // 5)
        val_idx.extend(cls_indices[:n_val].tolist())
        train_idx.extend(cls_indices[n_val:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return (
        features[train_idx],
        labels[train_idx],
        features[val_idx],
        labels[val_idx],
    )


def _evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    cls_correct = np.zeros(len(CLASSES), dtype=np.int64)
    cls_total = np.zeros(len(CLASSES), dtype=np.int64)

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += float(loss.item()) * yb.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == yb).sum().item())
            total_samples += int(yb.size(0))

            for cls_idx in range(len(CLASSES)):
                mask = yb == cls_idx
                cnt = int(mask.sum().item())
                if cnt > 0:
                    cls_total[cls_idx] += cnt
                    cls_correct[cls_idx] += int((preds[mask] == yb[mask]).sum().item())

    avg_loss = total_loss / max(1, total_samples)
    overall_acc = correct / max(1, total_samples)
    per_class = {
        CLASSES[i]: (float(cls_correct[i] / cls_total[i]) if cls_total[i] > 0 else 0.0)
        for i in range(len(CLASSES))
    }
    return avg_loss, overall_acc, per_class


def train_model(
    epochs: int = 20,
    batch_size: int = 32,
    use_synthetic: bool = True,
    data_dir: Path | None = None,
    output_ckpt: Path = DEFAULT_CHECKPOINT_PATH,
    num_samples_per_class: int = 300,
    lr: float = 1.5e-3,
    seed: int = 42,
    val_fraction: float = 0.2,
) -> dict[str, Any]:
    """Train AcousticMobileNet and save best checkpoint + metrics JSON."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device, device_note = resolve_training_device(prefer_cuda=True)
    print(f"[device] {device_note}")

    if data_dir is not None and not use_synthetic:
        waveforms, labels = _load_wav_directory_dataset(Path(data_dir))
        data_source = f"directory:{data_dir}"
    else:
        waveforms, labels = generate_synthetic_dataset(
            num_samples_per_class=num_samples_per_class,
            seed=seed,
            sr=SAMPLE_RATE,
        )
        data_source = f"synthetic:{num_samples_per_class}_per_class"

    spectrograms = extract_batch_spectrograms(waveforms, sr=SAMPLE_RATE)
    x_train, y_train, x_val, y_val = _stratified_split(
        spectrograms, labels, val_fraction=val_fraction, seed=seed
    )

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = AcousticMobileNet(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=1e-5
    )

    ckpt_path = Path(output_ckpt)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = ckpt_path.parent / DEFAULT_METRICS_PATH.name

    best_score = -1.0
    best_epoch = 0
    best_val_acc = 0.0
    best_per_class: dict[str, float] = {c: 0.0 for c in CLASSES}
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * yb.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += int((preds == yb).sum().item())
            train_total += int(yb.size(0))

        scheduler.step()

        train_loss = running_loss / max(1, train_total)
        train_acc = train_correct / max(1, train_total)
        val_loss, val_acc, per_class = _evaluate_model(model, val_loader, criterion, device)
        shahed_acc = per_class["Shahed_Piston_UAV"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "train_acc": round(train_acc, 5),
                "val_loss": round(val_loss, 5),
                "val_acc": round(val_acc, 5),
                "shahed_val_acc": round(shahed_acc, 5),
                "lr": round(float(optimizer.param_groups[0]["lr"]), 7),
            }
        )

        # Favor balanced validation accuracy with slight tie-breaker on Shahed recall
        composite = val_acc + 0.05 * shahed_acc
        if composite >= best_score:
            best_score = composite
            best_epoch = epoch
            best_val_acc = val_acc
            best_per_class = per_class
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "classes": CLASSES,
                    "num_classes": len(CLASSES),
                    "val_accuracy": val_acc,
                    "per_class_accuracy": per_class,
                    "shahed_val_accuracy": shahed_acc,
                    "device": str(device),
                    "seed": seed,
                    "dsp_config": {
                        "sample_rate": SAMPLE_RATE,
                        "n_fft": N_FFT,
                        "hop_length": HOP_LENGTH,
                        "num_frames": NUM_FRAMES,
                        "n_mels": N_MELS,
                        "fmin": FMIN,
                        "fmax": FMAX,
                    },
                },
                ckpt_path,
            )

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"shahed_acc={shahed_acc:.4f}"
        )

    metrics = {
        "data_source": data_source,
        "device": str(device),
        "device_note": device_note,
        "epochs": epochs,
        "batch_size": batch_size,
        "num_train_samples": int(x_train.shape[0]),
        "num_val_samples": int(x_val.shape[0]),
        "best_epoch": best_epoch,
        "best_val_accuracy": round(best_val_acc, 6),
        "shahed_val_accuracy": round(best_per_class["Shahed_Piston_UAV"], 6),
        "per_class_accuracy": {k: round(v, 6) for k, v in best_per_class.items()},
        "checkpoint_path": str(ckpt_path),
        "history": history,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AcousticMobileNet UAV classifier.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--synthetic", action="store_true", help="Train on synthetic dataset.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Optional WAV dataset dir.")
    parser.add_argument("--output-ckpt", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--num-samples-per-class", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    use_synth = args.synthetic or args.data_dir is None
    metrics = train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_synthetic=use_synth,
        data_dir=args.data_dir,
        output_ckpt=args.output_ckpt,
        num_samples_per_class=args.num_samples_per_class,
        lr=args.lr,
        seed=args.seed,
    )
    print(
        f"Best epoch {metrics['best_epoch']}: val_acc={metrics['best_val_accuracy']:.4f}, "
        f"Shahed_Piston_UAV={metrics['shahed_val_accuracy']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
