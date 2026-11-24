"""Cross-stack DSP and ONNX parity helper for Panotti.

Generates deterministic 16 kHz Shahed-136 comb PCM samples, computes the Python
reference Log-Mel spectrogram, runs ONNX Runtime inference on the quantized
uav_detector_int8.onnx model, and emits JSON for TypeScript parity comparison.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_BUILDER_DIR = REPO_ROOT / "pipelines" / "dataset_builder"
sys.path.insert(0, str(DATASET_BUILDER_DIR))

from config import CLASSES  # noqa: E402
from feature_extraction import extract_log_mel_spectrogram  # noqa: E402


def generate_deterministic_shahed_comb(sample_rate: int = 16000, f0: float = 92.5) -> np.ndarray:
    t = np.arange(sample_rate, dtype=np.float64) / float(sample_rate)
    sig = np.zeros(sample_rate, dtype=np.float64)
    k = 1
    while k * f0 <= 4000.0:
        freq = k * f0
        amp = (1.0 / (k ** 0.78)) * np.exp(-freq / 3800.0)
        sig += amp * np.sin(2.0 * np.pi * freq * t + 0.37 * k)
        k += 1
    sig *= 0.28 * (1.0 + 0.08 * np.sin(2.0 * np.pi * 18.5 * t))
    return sig.astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def main() -> int:
    pcm = generate_deterministic_shahed_comb()
    spec = extract_log_mel_spectrogram(pcm)  # [1, 128, 100]
    model_path = REPO_ROOT / "apps" / "web" / "public" / "models" / "uav_detector_int8.onnx"
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    logits = sess.run(["class_logits"], {"input_spectrogram": spec[np.newaxis, ...]})[0][0]
    probs = softmax(logits)
    top_idx = int(np.argmax(probs))

    out = {
        "pcm": pcm.tolist(),
        "spectrogram": spec.reshape(-1).tolist(),
        "logits": logits.tolist(),
        "probabilities": {cls: float(probs[i]) for i, cls in enumerate(CLASSES)},
        "topClass": CLASSES[top_idx],
        "confidence": float(probs[top_idx]),
    }

    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(out), encoding="utf-8")
    else:
        print(json.dumps({
            "topClass": out["topClass"],
            "confidence": out["confidence"],
            "modelBytes": model_path.stat().st_size,
        }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
