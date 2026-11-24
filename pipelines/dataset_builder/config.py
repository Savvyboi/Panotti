"""Shared configuration constants, DSP parameters, class labels, and paths."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

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

CLASS_OSINT_QUERIES: dict[str, list[str]] = {
    "Shahed_Piston_UAV": [
        "shahed 136 drone sound",
        "Limbach L550E engine",
        "Geran 2 drone flyby",
        "DLE 55 gas RC airplane flyby",
        "DLE 111 boxer RC engine sound",
        "DA 100 2 stroke RC plane flyby",
        "3W 110 boxer engine sound",
        "Saito FG 40 4 stroke RC plane sound",
        "Rotax 912 ultralight flyby sound",
        "Vittorazi Moster 185 paramotor sound",
        "2 stroke paramotor flyby close",
        "OS 49 PI Wankel rotary RC engine sound",
        "Wankel rotary engine UAV test",
    ],
    "Multirotor_UAV": [
        "DJI Mavic 3 Pro hovering sound test",
        "DJI Phantom 4 flying audio",
        "DJI Matrice 300 flying sound",
        "5 inch FPV drone flyby sound raw audio",
        "7 inch long range FPV drone sound",
        "FPV quadcopter buzzing sound effect",
        "DJI Agras T40 drone flying sound",
        "heavy lift octocopter drone sound",
    ],
    "FixedWing_Electric_UAV": [
        "FPV flying wing pusher motor sound",
        "RC flying wing flyby raw audio",
        "electric pusher RC plane sound",
        "fixed wing electric UAV takeoff flight sound",
        "brushless RC airplane flyby sound",
    ],
    "Ground_Combustion_Noise": [
        "50cc 2 stroke moped exhaust sound clip",
        "Vespa 2 stroke scooter revving sound",
        "Stihl 2 stroke chainsaw cutting sound",
        "2 stroke weed whacker string trimmer sound",
        "2 stroke backpack leaf blower sound clip",
        "gas lawnmower running sound effect",
        "portable gas generator running sound clip",
        "2 stroke dirt bike idle rev sound clip",
    ],
    "Environmental_Background": [
        "strong wind howling sound effect short",
        "heavy rain thunder sound effect short",
        "highway cars passing sound effect",
        "city street traffic ambient clip",
        "air raid siren Ukraine sound clip",
        "civil defense siren sound effect",
    ],
}

EXCLUDED_TITLE_KEYWORDS: tuple[str, ...] = (
    "song",
    "music video",
    "official video",
    "ocarina",
    "sound of nature",
    "podcast",
    "interview",
    "reaction",
    "documentary",
    "10 hours",
    "8 hours",
    "sleep sound",
    "white noise for sleeping",
    "what you need to know",
    "how to build",
    "maintenance",
    "hints & tips",
    "tutorial",
    "startup sound",
    "wont stop beeping",
    "earrape",
    "sound like gas",
    "suspense drone",
    "spitfire",
    "jet cat",
)

CLASS_CLEANING_PROFILES: dict[str, dict[str, Any]] = {
    "Shahed_Piston_UAV": {
        "comb_min_hz": 50.0,
        "comb_max_hz": 300.0,
        "comb_threshold": 0.22,
        "require_comb": True,
        "apply_siren_filter": True,
    },
    "Multirotor_UAV": {
        "comb_min_hz": 80.0,
        "comb_max_hz": 900.0,
        "comb_threshold": 0.20,
        "require_comb": True,
        "apply_siren_filter": True,
    },
    "FixedWing_Electric_UAV": {
        "comb_min_hz": 100.0,
        "comb_max_hz": 1200.0,
        "comb_threshold": 0.18,
        "require_comb": True,
        "apply_siren_filter": True,
    },
    "Ground_Combustion_Noise": {
        "comb_min_hz": 40.0,
        "comb_max_hz": 350.0,
        "comb_threshold": 0.20,
        "require_comb": True,
        "apply_siren_filter": True,
    },
    "Environmental_Background": {
        "comb_min_hz": 50.0,
        "comb_max_hz": 300.0,
        "comb_threshold": 0.25,
        "require_comb": False,
        "apply_siren_filter": False,
    },
}

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
