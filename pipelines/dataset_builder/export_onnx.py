"""Export AcousticMobileNet to FP32 ONNX (opset 17), dynamically quantize to INT8, and verify parity."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from config import (
    CLASSES,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_FP32_ONNX_PATH,
    DEFAULT_INT8_ONNX_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_METADATA_PATH,
    EPS,
    FMAX,
    FMIN,
    HOP_LENGTH,
    MAX_ONNX_BYTES,
    N_FFT,
    N_MELS,
    NUM_FRAMES,
    ONNX_INPUT_NAME,
    ONNX_OPSET,
    ONNX_OUTPUT_NAME,
    SAMPLE_RATE,
)
from model import AcousticMobileNet


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=-1, keepdims=True)


def verify_onnx_parity(
    model: AcousticMobileNet,
    fp32_onnx_path: Path,
    int8_onnx_path: Path,
    num_test_samples: int = 8,
    seed: int = 123,
    fp32_atol: float = 1e-4,
    int8_prob_atol: float = 0.15,
) -> dict[str, Any]:
    """Verify ONNX model structure and output parity against PyTorch."""
    from feature_extraction import extract_batch_spectrograms
    from synthetic_dataset import generate_synthetic_dataset

    onnx_model = onnx.load(str(int8_onnx_path))
    onnx.checker.check_model(onnx_model)

    rng = np.random.default_rng(seed)
    waves, _ = generate_synthetic_dataset(num_samples_per_class=2, seed=seed)
    synth_specs = extract_batch_spectrograms(waves, sr=SAMPLE_RATE)
    rand_specs = rng.normal(0.0, 1.0, size=(max(2, num_test_samples // 2), 1, N_MELS, NUM_FRAMES)).astype(np.float32)
    test_np = np.concatenate([synth_specs, rand_specs], axis=0).astype(np.float32)

    model.eval()
    with torch.no_grad():
        pt_logits = model(torch.from_numpy(test_np)).cpu().numpy()

    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3

    fp32_sess = ort.InferenceSession(str(fp32_onnx_path), sess_options=sess_opts, providers=["CPUExecutionProvider"])
    fp32_logits = fp32_sess.run([ONNX_OUTPUT_NAME], {ONNX_INPUT_NAME: test_np})[0]

    int8_sess = ort.InferenceSession(str(int8_onnx_path), sess_options=sess_opts, providers=["CPUExecutionProvider"])
    int8_logits = int8_sess.run([ONNX_OUTPUT_NAME], {ONNX_INPUT_NAME: test_np})[0]

    fp32_max_diff = float(np.max(np.abs(pt_logits - fp32_logits)))
    int8_max_diff = float(np.max(np.abs(pt_logits - int8_logits)))

    pt_probs = _softmax(pt_logits)
    int8_probs = _softmax(int8_logits)
    int8_prob_max_diff = float(np.max(np.abs(pt_probs - int8_probs)))

    pt_argmax = np.argmax(pt_logits, axis=1)
    fp32_argmax = np.argmax(fp32_logits, axis=1)
    int8_argmax = np.argmax(int8_logits, axis=1)

    if fp32_max_diff > fp32_atol:
        raise AssertionError(f"FP32 ONNX max diff {fp32_max_diff:.6f} exceeded {fp32_atol}")
    if not np.array_equal(pt_argmax, fp32_argmax):
        raise AssertionError("FP32 ONNX argmax diverged from PyTorch")

    # Verify quantized model preserves batching shape and sensible probability parity
    assert int8_logits.shape == (test_np.shape[0], len(CLASSES))
    if int8_prob_max_diff > int8_prob_atol:
        raise AssertionError(
            f"INT8 ONNX softmax probability max diff {int8_prob_max_diff:.6f} exceeded {int8_prob_atol}"
        )

    argmax_match_rate = float(np.mean(pt_argmax == int8_argmax))
    return {
        "fp32_max_abs_diff": round(fp32_max_diff, 8),
        "int8_max_abs_diff": round(int8_max_diff, 6),
        "int8_prob_max_abs_diff": round(int8_prob_max_diff, 6),
        "argmax_match_rate": round(argmax_match_rate, 4),
    }


def export_quantized_onnx(
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    fp32_output: Path = DEFAULT_FP32_ONNX_PATH,
    int8_output: Path = DEFAULT_INT8_ONNX_PATH,
    metadata_output: Path = DEFAULT_MODEL_METADATA_PATH,
    metrics_path: Path | None = None,
) -> dict[str, Any]:
    """Export trained checkpoint to FP32 ONNX opset 17, dynamically quantize to INT8, and write metadata."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    classes = list(checkpoint.get("classes", CLASSES))

    model = AcousticMobileNet(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    fp32_path = Path(fp32_output)
    fp32_path.parent.mkdir(parents=True, exist_ok=True)
    int8_path = Path(int8_output)
    int8_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = Path(metadata_output)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, 1, N_MELS, NUM_FRAMES, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        str(fp32_path),
        export_params=True,
        opset_version=ONNX_OPSET,
        do_constant_folding=True,
        dynamo=False,
        input_names=[ONNX_INPUT_NAME],
        output_names=[ONNX_OUTPUT_NAME],
        dynamic_axes={
            ONNX_INPUT_NAME: {0: "batch_size"},
            ONNX_OUTPUT_NAME: {0: "batch_size"},
        },
    )

    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QUInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )

    int8_bytes = int8_path.stat().st_size
    if int8_bytes >= MAX_ONNX_BYTES:
        raise AssertionError(
            f"Quantized ONNX model size {int8_bytes} bytes exceeds 8 MB limit ({MAX_ONNX_BYTES})."
        )

    parity = verify_onnx_parity(model, fp32_path, int8_path)

    resolved_metrics_path = Path(metrics_path) if metrics_path else (ckpt_path.parent / DEFAULT_METRICS_PATH.name)
    val_metrics: dict[str, Any] = {}
    if resolved_metrics_path.exists():
        val_metrics = json.loads(resolved_metrics_path.read_text(encoding="utf-8"))
    else:
        val_metrics = {
            "best_val_accuracy": checkpoint.get("val_accuracy"),
            "shahed_val_accuracy": checkpoint.get("shahed_val_accuracy"),
            "per_class_accuracy": checkpoint.get("per_class_accuracy", {}),
        }

    metadata: dict[str, Any] = {
        "model_name": "AcousticMobileNetV3Small",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "classes": classes,
        "num_classes": len(classes),
        "input_name": ONNX_INPUT_NAME,
        "output_name": ONNX_OUTPUT_NAME,
        "input_shape": [1, 1, N_MELS, NUM_FRAMES],
        "opset_version": ONNX_OPSET,
        "quantization": "dynamic_quint8",
        "fp32_size_bytes": fp32_path.stat().st_size,
        "int8_size_bytes": int8_bytes,
        "dsp": {
            "sample_rate": SAMPLE_RATE,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "num_frames": NUM_FRAMES,
            "n_mels": N_MELS,
            "fmin": FMIN,
            "fmax": FMAX,
            "eps": EPS,
        },
        "validation_metrics": {
            "data_source": val_metrics.get("data_source", checkpoint.get("data_source")),
            "split_strategy": val_metrics.get("split_strategy", checkpoint.get("split_strategy")),
            "best_val_accuracy": val_metrics.get("best_val_accuracy"),
            "val_video_accuracy": val_metrics.get("val_video_accuracy", checkpoint.get("val_video_accuracy")),
            "macro_f1": val_metrics.get("macro_f1", checkpoint.get("macro_f1")),
            "shahed_val_accuracy": val_metrics.get("shahed_val_accuracy"),
            "per_class_accuracy": val_metrics.get("per_class_accuracy", {}),
        },
        "parity": parity,
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export and dynamically quantize AcousticMobileNet to ONNX.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--fp32-output", type=Path, default=DEFAULT_FP32_ONNX_PATH)
    parser.add_argument("--int8-output", type=Path, default=DEFAULT_INT8_ONNX_PATH)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_MODEL_METADATA_PATH)
    parser.add_argument("--metrics-path", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metadata = export_quantized_onnx(
        checkpoint_path=args.checkpoint,
        fp32_output=args.fp32_output,
        int8_output=args.int8_output,
        metadata_output=args.metadata_output,
        metrics_path=args.metrics_path,
    )
    print(
        f"Exported INT8 ONNX to {args.int8_output} "
        f"({metadata['int8_size_bytes']} bytes, {metadata['int8_size_bytes'] / (1024 * 1024):.2f} MB)"
    )
    print(f"Parity check: {metadata['parity']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
