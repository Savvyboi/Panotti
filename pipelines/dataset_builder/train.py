"""Training pipeline for AcousticMobileNet with video-grouped split, leakage guard, and CPU/CUDA auto-check."""

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


def extract_source_group_id(wav_path: Path) -> str:
    """Extract source recording ID from <vid>_frame_0001[_aug_00].wav to prevent leakage."""
    return wav_path.stem.split("_frame_")[0]


def load_wav_directory_with_groups(
    data_dir: Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load 1-second mono 16kHz WAVs arranged under data_dir/<class_name>/**/*.wav with group IDs."""
    clips: list[np.ndarray] = []
    labels: list[int] = []
    group_ids: list[str] = []

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
                arr = signal.resample_poly(
                    arr, SAMPLE_RATE // gcd, int(orig_sr) // gcd
                ).astype(np.float32)

            group_id = extract_source_group_id(wav_path)
            if arr.size < NUM_SAMPLES:
                arr = np.pad(arr, (0, NUM_SAMPLES - arr.size))
                clips.append(arr)
                labels.append(class_idx)
                group_ids.append(group_id)
            else:
                for start in range(0, arr.size - NUM_SAMPLES + 1, NUM_SAMPLES):
                    clips.append(arr[start : start + NUM_SAMPLES])
                    labels.append(class_idx)
                    group_ids.append(group_id)

    if not clips:
        raise FileNotFoundError(f"No WAV samples found in class subdirectories under {data_dir}.")

    return np.stack(clips, axis=0).astype(np.float32), np.asarray(labels, dtype=np.int64), group_ids


def _load_wav_directory_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load 1-second mono 16kHz WAVs arranged under data_dir/<class_name>/*.wav."""
    clips, labels, _ = load_wav_directory_with_groups(data_dir)
    return clips, labels


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


def grouped_stratified_split(
    labels: np.ndarray,
    group_ids: list[str],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict[int, set[str]], dict[int, set[str]]]:
    """Split clip indices strictly by source video group_id per class to prevent leakage."""
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    train_groups_by_cls: dict[int, set[str]] = {}
    val_groups_by_cls: dict[int, set[str]] = {}

    group_arr = np.asarray(group_ids, dtype=object)

    for cls_idx in range(len(CLASSES)):
        cls_mask = np.where(labels == cls_idx)[0]
        if cls_mask.size == 0:
            train_groups_by_cls[cls_idx] = set()
            val_groups_by_cls[cls_idx] = set()
            continue

        unique_groups = sorted({str(group_arr[i]) for i in cls_mask})
        rng.shuffle(unique_groups)

        if len(unique_groups) == 1:
            # Degenerate single-group fallback: split frames deterministically
            grp = unique_groups[0]
            train_groups_by_cls[cls_idx] = {grp}
            val_groups_by_cls[cls_idx] = set()
            shuffled = cls_mask.copy()
            rng.shuffle(shuffled)
            n_val = max(1, int(round(shuffled.size * val_fraction)))
            if n_val >= shuffled.size:
                n_val = max(1, shuffled.size // 5)
            val_idx.extend(shuffled[:n_val].tolist())
            train_idx.extend(shuffled[n_val:].tolist())
            continue

        n_val_groups = max(1, int(round(len(unique_groups) * val_fraction)))
        n_val_groups = min(n_val_groups, len(unique_groups) - 1)

        val_grps = set(unique_groups[:n_val_groups])
        train_grps = set(unique_groups[n_val_groups:])
        assert val_grps.isdisjoint(train_grps)

        val_groups_by_cls[cls_idx] = val_grps
        train_groups_by_cls[cls_idx] = train_grps

        for idx in cls_mask.tolist():
            if str(group_arr[idx]) in val_grps:
                val_idx.append(idx)
            else:
                train_idx.append(idx)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(val_idx, dtype=np.int64),
        train_groups_by_cls,
        val_groups_by_cls,
    )


def apply_spec_augment(
    x_batch: torch.Tensor,
    max_freq_mask: int = 12,
    max_time_mask: int = 12,
    p: float = 0.5,
) -> torch.Tensor:
    """Apply lightweight SpecAugment (1 frequency mask + 1 time mask) on training batches."""
    if p <= 0.0:
        return x_batch
    out = x_batch.clone()
    bsz, _, n_mels, n_frames = out.shape
    for i in range(bsz):
        if float(torch.rand(1).item()) >= p:
            continue
        f_w = int(torch.randint(1, min(max_freq_mask, n_mels) + 1, (1,)).item())
        f_0 = int(torch.randint(0, max(1, n_mels - f_w + 1), (1,)).item())
        out[i, :, f_0 : f_0 + f_w, :] = 0.0

        t_w = int(torch.randint(1, min(max_time_mask, n_frames) + 1, (1,)).item())
        t_0 = int(torch.randint(0, max(1, n_frames - t_w + 1), (1,)).item())
        out[i, :, :, t_0 : t_0 + t_w] = 0.0
    return out


def _compute_macro_f1(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> float:
    f1s: list[float] = []
    for cls_idx in range(num_classes):
        tp = int(np.sum((preds == cls_idx) & (targets == cls_idx)))
        fp = int(np.sum((preds == cls_idx) & (targets != cls_idx)))
        fn = int(np.sum((preds != cls_idx) & (targets == cls_idx)))
        denom = 2 * tp + fp + fn
        if denom > 0:
            f1s.append((2.0 * tp) / float(denom))
    return float(np.mean(f1s)) if f1s else 0.0


def _evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, dict[str, float], float, dict[str, int]]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    cls_correct = np.zeros(len(CLASSES), dtype=np.int64)
    cls_total = np.zeros(len(CLASSES), dtype=np.int64)
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

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
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())

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
    per_class_counts = {CLASSES[i]: int(cls_total[i]) for i in range(len(CLASSES))}
    preds_cat = np.concatenate(all_preds, axis=0) if all_preds else np.empty(0, dtype=np.int64)
    targets_cat = np.concatenate(all_targets, axis=0) if all_targets else np.empty(0, dtype=np.int64)
    macro_f1 = _compute_macro_f1(preds_cat, targets_cat, len(CLASSES))
    return avg_loss, overall_acc, per_class, macro_f1, per_class_counts


def _cap_frames_per_video(
    waves: np.ndarray,
    labels: np.ndarray,
    group_ids: list[str],
    max_per_video: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Deterministically subsample up to max_per_video evenly spaced frames per (class, group_id)."""
    if max_per_video <= 0:
        return waves, labels, group_ids

    buckets: dict[tuple[int, str], list[int]] = {}
    for idx, (lbl, grp) in enumerate(zip(labels.tolist(), group_ids)):
        buckets.setdefault((int(lbl), grp), []).append(idx)

    keep_indices: list[int] = []
    for key in sorted(buckets.keys()):
        idxs = buckets[key]
        if len(idxs) <= max_per_video:
            keep_indices.extend(idxs)
        else:
            sel = np.linspace(0, len(idxs) - 1, max_per_video, dtype=np.int64)
            keep_indices.extend(idxs[int(s)] for s in sel)

    keep_indices.sort()
    keep_arr = np.asarray(keep_indices, dtype=np.int64)
    return waves[keep_arr], labels[keep_arr], [group_ids[int(i)] for i in keep_arr]


def apply_cyclic_time_roll(
    x_batch: torch.Tensor,
    max_shift: int = 20,
    p: float = 0.5,
) -> torch.Tensor:
    """Variance-preserving cyclic roll along the spectrogram time axis during training."""
    if p <= 0.0 or max_shift <= 0:
        return x_batch
    out = x_batch.clone()
    for i in range(out.size(0)):
        if float(torch.rand(1).item()) >= p:
            continue
        shift = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
        if shift != 0:
            out[i] = torch.roll(out[i], shifts=shift, dims=-1)
    return out


def train_model(
    epochs: int = 20,
    batch_size: int = 32,
    use_synthetic: bool = True,
    data_dir: Path | None = None,
    augmented_dir: Path | None = None,
    mix_synthetic: int = 0,
    max_frames_per_video: int = 0,
    output_ckpt: Path = DEFAULT_CHECKPOINT_PATH,
    num_samples_per_class: int = 300,
    lr: float = 1.5e-3,
    seed: int = 42,
    val_fraction: float = 0.2,
    label_smoothing: float = 0.05,
    weight_decay: float = 5e-4,
    spec_augment: bool = True,
    pretrained: bool = False,
    dropout: float = 0.2,
    mixup_alpha: float = 0.0,
) -> dict[str, Any]:
    """Train AcousticMobileNet and save best checkpoint + metrics JSON."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device, device_note = resolve_training_device(prefer_cuda=True)
    print(f"[device] {device_note}", flush=True)

    train_video_count = 0
    val_video_count = 0
    per_class_val_counts: dict[str, int] = {c: 0 for c in CLASSES}
    val_group_ids: list[str] | None = None

    if data_dir is not None and not use_synthetic:
        clean_waves, clean_labels, clean_groups = load_wav_directory_with_groups(Path(data_dir))
        if max_frames_per_video > 0:
            clean_waves, clean_labels, clean_groups = _cap_frames_per_video(
                clean_waves, clean_labels, clean_groups, max_frames_per_video
            )
        train_idx, val_idx, train_groups_by_cls, val_groups_by_cls = grouped_stratified_split(
            clean_labels, clean_groups, val_fraction=val_fraction, seed=seed
        )
        train_video_count = sum(len(s) for s in train_groups_by_cls.values())
        val_video_count = sum(len(s) for s in val_groups_by_cls.values())
        val_group_ids = [clean_groups[int(i)] for i in val_idx]

        all_train_vids = set().union(*train_groups_by_cls.values()) if train_groups_by_cls else set()
        all_val_vids = set().union(*val_groups_by_cls.values()) if val_groups_by_cls else set()
        cross_overlap = all_train_vids & all_val_vids
        if cross_overlap:
            print(f"[warning] Cross-class shared video IDs detected between train/val: {sorted(cross_overlap)}", flush=True)

        clean_specs = extract_batch_spectrograms(clean_waves, sr=SAMPLE_RATE)
        x_val = clean_specs[val_idx]
        y_val = clean_labels[val_idx]

        train_spec_parts = [clean_specs[train_idx]]
        train_label_parts = [clean_labels[train_idx]]
        aug_kept = 0

        if augmented_dir is not None:
            aug_waves, aug_labels, aug_groups = load_wav_directory_with_groups(Path(augmented_dir))
            if max_frames_per_video > 0:
                aug_waves, aug_labels, aug_groups = _cap_frames_per_video(
                    aug_waves, aug_labels, aug_groups, max_frames_per_video * 3
                )
            keep_mask = np.zeros(aug_labels.shape[0], dtype=bool)
            for idx in range(aug_labels.shape[0]):
                cls_idx = int(aug_labels[idx])
                grp = aug_groups[idx]
                if grp in train_groups_by_cls.get(cls_idx, set()) and grp not in val_groups_by_cls.get(cls_idx, set()):
                    keep_mask[idx] = True

            if np.any(keep_mask):
                aug_specs = extract_batch_spectrograms(aug_waves[keep_mask], sr=SAMPLE_RATE)
                train_spec_parts.append(aug_specs)
                train_label_parts.append(aug_labels[keep_mask])
                aug_kept = int(np.sum(keep_mask))

        if mix_synthetic > 0:
            syn_waves, syn_labels = generate_synthetic_dataset(
                num_samples_per_class=mix_synthetic,
                seed=seed + 1000,
                sr=SAMPLE_RATE,
            )
            syn_specs = extract_batch_spectrograms(syn_waves, sr=SAMPLE_RATE)
            train_spec_parts.append(syn_specs)
            train_label_parts.append(syn_labels)

        x_train = np.concatenate(train_spec_parts, axis=0).astype(np.float32, copy=False)
        y_train = np.concatenate(train_label_parts, axis=0).astype(np.int64, copy=False)
        split_strategy = "grouped_by_source_video_id"
        data_source = (
            f"directory:{data_dir}"
            + (f"+aug:{augmented_dir}({aug_kept})" if augmented_dir else "")
            + (f"+syn:{mix_synthetic}_per_class" if mix_synthetic > 0 else "")
        )
    else:
        waveforms, labels = generate_synthetic_dataset(
            num_samples_per_class=num_samples_per_class,
            seed=seed,
            sr=SAMPLE_RATE,
        )
        spectrograms = extract_batch_spectrograms(waveforms, sr=SAMPLE_RATE)
        x_train, y_train, x_val, y_val = _stratified_split(
            spectrograms, labels, val_fraction=val_fraction, seed=seed
        )
        split_strategy = "stratified_synthetic"
        data_source = f"synthetic:{num_samples_per_class}_per_class"

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # Inverse-frequency class weighting for imbalanced real-world recordings
    cls_counts = np.bincount(y_train, minlength=len(CLASSES)).astype(np.float64)
    raw_weights = np.where(cls_counts > 0, len(y_train) / (len(CLASSES) * np.maximum(cls_counts, 1.0)), 1.0)
    norm_weights = raw_weights / max(float(np.mean(raw_weights)), 1e-12)
    weight_tensor = torch.tensor(norm_weights, dtype=torch.float32, device=device)

    model = AcousticMobileNet(num_classes=len(CLASSES), dropout=dropout, pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)
    val_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=1e-5
    )

    ckpt_path = Path(output_ckpt)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = ckpt_path.parent / DEFAULT_METRICS_PATH.name

    best_score = -1.0
    best_epoch = 0
    best_val_acc = 0.0
    best_macro_f1 = 0.0
    best_per_class: dict[str, float] = {c: 0.0 for c in CLASSES}
    history: list[dict[str, Any]] = []
    rng_mix = np.random.default_rng(seed + 77)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            if spec_augment:
                xb = apply_cyclic_time_roll(xb, max_shift=20, p=0.5)
                xb = apply_spec_augment(xb, max_freq_mask=10, max_time_mask=10, p=0.4)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)

            if mixup_alpha > 0.0 and xb.size(0) >= 2 and float(rng_mix.random()) < 0.5:
                lam = float(rng_mix.beta(mixup_alpha, mixup_alpha))
                perm = torch.randperm(xb.size(0), device=device)
                mixed_x = lam * xb + (1.0 - lam) * xb[perm]
                logits = model(mixed_x)
                loss = lam * criterion(logits, yb) + (1.0 - lam) * criterion(logits, yb[perm])
            else:
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
        val_loss, val_acc, per_class, macro_f1, per_class_val_counts = _evaluate_model(
            model, val_loader, val_criterion, device
        )
        shahed_acc = per_class["Shahed_Piston_UAV"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "train_acc": round(train_acc, 5),
                "val_loss": round(val_loss, 5),
                "val_acc": round(val_acc, 5),
                "macro_f1": round(macro_f1, 5),
                "shahed_val_acc": round(shahed_acc, 5),
                "lr": round(float(optimizer.param_groups[0]["lr"]), 7),
            }
        )

        # Favor balanced macro F1 + accuracy with slight tie-breaker on Shahed recall
        composite = 0.5 * val_acc + 0.5 * macro_f1 + 0.03 * shahed_acc
        if composite >= best_score:
            best_score = composite
            best_epoch = epoch
            best_val_acc = val_acc
            best_macro_f1 = macro_f1
            best_per_class = per_class
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "classes": CLASSES,
                    "num_classes": len(CLASSES),
                    "val_accuracy": val_acc,
                    "macro_f1": macro_f1,
                    "per_class_accuracy": per_class,
                    "shahed_val_accuracy": shahed_acc,
                    "data_source": data_source,
                    "split_strategy": split_strategy,
                    "pretrained": pretrained,
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
            f"macro_f1={macro_f1:.4f} shahed_acc={shahed_acc:.4f}",
            flush=True,
        )

    val_video_accuracy: float | None = None
    if val_group_ids is not None and ckpt_path.exists():
        best_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        eval_model = AcousticMobileNet(num_classes=len(CLASSES), dropout=dropout, pretrained=False).to(device)
        eval_model.load_state_dict(best_ckpt["model_state_dict"])
        eval_model.eval()
        all_probs: list[np.ndarray] = []
        with torch.no_grad():
            for xb, _ in val_loader:
                probs = torch.softmax(eval_model(xb.to(device)), dim=1).cpu().numpy()
                all_probs.append(probs)
        probs_cat = np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, len(CLASSES)))
        vid_buckets: dict[tuple[int, str], list[np.ndarray]] = {}
        for prob_vec, lbl, grp in zip(probs_cat, y_val.tolist(), val_group_ids):
            vid_buckets.setdefault((int(lbl), grp), []).append(prob_vec)
        if vid_buckets:
            vid_correct = sum(
                1 for (lbl, _), vecs in vid_buckets.items() if int(np.argmax(np.mean(vecs, axis=0))) == lbl
            )
            val_video_accuracy = round(vid_correct / float(len(vid_buckets)), 6)
            best_ckpt["val_video_accuracy"] = val_video_accuracy
            torch.save(best_ckpt, ckpt_path)

    metrics = {
        "data_source": data_source,
        "split_strategy": split_strategy,
        "pretrained": pretrained,
        "dropout": dropout,
        "mixup_alpha": mixup_alpha,
        "train_video_count": train_video_count,
        "val_video_count": val_video_count,
        "val_video_accuracy": val_video_accuracy,
        "per_class_val_counts": per_class_val_counts,
        "device": str(device),
        "device_note": device_note,
        "epochs": epochs,
        "batch_size": batch_size,
        "num_train_samples": int(x_train.shape[0]),
        "num_val_samples": int(x_val.shape[0]),
        "best_epoch": best_epoch,
        "best_val_accuracy": round(best_val_acc, 6),
        "macro_f1": round(best_macro_f1, 6),
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
    parser.add_argument("--data-dir", type=Path, default=None, help="Optional clean WAV dataset dir.")
    parser.add_argument(
        "--augmented-dir",
        type=Path,
        default=None,
        help="Optional augmented WAV dir (strictly restricted to training video IDs).",
    )
    parser.add_argument(
        "--mix-synthetic",
        type=int,
        default=0,
        help="Optional synthetic samples per class mixed strictly into training split.",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Initialize MobileNetV3-Small backbone with ImageNet pretrained weights.",
    )
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--max-frames-per-video", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
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
        augmented_dir=args.augmented_dir,
        mix_synthetic=args.mix_synthetic,
        max_frames_per_video=args.max_frames_per_video,
        output_ckpt=args.output_ckpt,
        num_samples_per_class=args.num_samples_per_class,
        lr=args.lr,
        seed=args.seed,
        weight_decay=args.weight_decay,
        pretrained=args.pretrained,
        dropout=args.dropout,
        mixup_alpha=args.mixup_alpha,
    )
    print(
        f"Best epoch {metrics['best_epoch']}: val_acc={metrics['best_val_accuracy']:.4f}, "
        f"macro_f1={metrics['macro_f1']:.4f}, Shahed_Piston_UAV={metrics['shahed_val_accuracy']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
