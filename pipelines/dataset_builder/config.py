"""Shared configuration constants, DSP parameters, class labels, and paths."""

from __future__ import annotations

import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Shared DSP contract (MUST match apps/web/src/audio/MelSpectrogramExtractor.ts)
# ---------------------------------------------------------------------------
SAMPLE_RATE: int = 16000
CLIP_SECONDS: float = 1.0
NUM_SAMPLES: int = 16000
N_FFT: int = 512
HOP_LENGTH: int = 160
NUM_FRAMES: int = 100
PADDED_LENGTH: int = NUM_SAMPLES + (N_FFT - HOP_LENGTH)  # 16352
N_MELS: int = 128
FMIN: float = 50.0
FMAX: float = 4000.0
EPS: float = 1e-6

# ---------------------------------------------------------------------------
# Exact 5 acoustic target classes (index 0..4)
# ---------------------------------------------------------------------------
CLASSES: list[str] = [
    "Shahed_Piston_UAV",
    "Multirotor_UAV",
    "FixedWing_Electric_UAV",
    "Ground_Combustion_Noise",
    "Environmental_Background",
]
CLASS_TO_IDX: dict[str, int] = {label: idx for idx, label in enumerate(CLASSES)}
UAV_CLASSES: set[str] = {
    "Shahed_Piston_UAV",
    "Multirotor_UAV",
    "FixedWing_Electric_UAV",
}

# ---------------------------------------------------------------------------
# Filtering & scraping defaults
# ---------------------------------------------------------------------------
DEFAULT_OSINT_QUERIES: list[str] = [
    "shahed 136 sound",
    "geran 2 audio",
    "moped drone flying sound",
    "paramotor flyby audio",
    "limbache L550E engine audio",
]

SIREN_GUNFIRE_LABELS: set[str] = {
    "Siren",
    "Civil defense siren",
    "Fire alarm",
    "Gunshot, gunfire",
}
SIREN_GUNFIRE_THRESHOLD: float = 0.30

# ---------------------------------------------------------------------------
# ONNX model contract & filesystem paths
# ---------------------------------------------------------------------------
ONNX_INPUT_NAME: str = "input_spectrogram"
ONNX_OUTPUT_NAME: str = "class_logits"
ONNX_OPSET: int = 17
MAX_ONNX_BYTES: int = 8 * 1024 * 1024

PACKAGE_DIR: Path = Path(__file__).resolve().parent
REPO_ROOT: Path = PACKAGE_DIR.parent.parent

DATA_DIR: Path = PACKAGE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
CLEAN_DATA_DIR: Path = DATA_DIR / "clean"
AUGMENTED_DATA_DIR: Path = DATA_DIR / "augmented"

CHECKPOINTS_DIR: Path = PACKAGE_DIR / "checkpoints"
DEFAULT_CHECKPOINT_PATH: Path = CHECKPOINTS_DIR / "best_model.pt"
DEFAULT_METRICS_PATH: Path = CHECKPOINTS_DIR / "metrics.json"
DEFAULT_FP32_ONNX_PATH: Path = PACKAGE_DIR / "uav_detector_fp32.onnx"

WEB_MODELS_DIR: Path = REPO_ROOT / "apps" / "web" / "public" / "models"
DEFAULT_INT8_ONNX_PATH: Path = WEB_MODELS_DIR / "uav_detector_int8.onnx"
DEFAULT_MODEL_METADATA_PATH: Path = WEB_MODELS_DIR / "model_metadata.json"


def resolve_ffmpeg_path(explicit_path: str | None = None) -> str | None:
    """Resolve an available ffmpeg executable path on the current machine."""
    if explicit_path:
        candidate = Path(explicit_path)
        if candidate.exists():
            return str(candidate)
        found = shutil.which(explicit_path)
        if found:
            return found

    win_default = Path(r"C:\FFmpeg\bin\ffmpeg.exe")
    if win_default.exists():
        return str(win_default)

    return shutil.which("ffmpeg")
