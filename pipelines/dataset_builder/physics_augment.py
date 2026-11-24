"""Physics-based acoustic augmentations: atmospheric absorption, Doppler, flutter, and SNR mixing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.interpolate import interp1d

from config import AUGMENTED_DATA_DIR, CLASSES, CLEAN_DATA_DIR, SAMPLE_RATE

SPEED_OF_SOUND_MPS: float = 343.0
DEFAULT_ALPHA0: float = 2.5e-9  # dB / (m * Hz^2) ISO 9613-1 inspired coefficient

CLASS_AUGMENT_PROFILES: dict[str, dict[str, Any]] = {
    "Shahed_Piston_UAV": {
        "doppler_prob": 0.75,
        "velocity_range": (20.0, 55.0),
        "min_dist_range": (80.0, 400.0),
        "distance_range": (50.0, 4000.0),
        "include_flutter": False,
        "snr_range": (3.0, 18.0),
        "gain_range": (0.8, 1.2),
    },
    "Multirotor_UAV": {
        "doppler_prob": 0.60,
        "velocity_range": (5.0, 30.0),
        "min_dist_range": (40.0, 250.0),
        "distance_range": (30.0, 1500.0),
        "include_flutter": True,
        "snr_range": (3.0, 18.0),
        "gain_range": (0.8, 1.2),
    },
    "FixedWing_Electric_UAV": {
        "doppler_prob": 0.75,
        "velocity_range": (15.0, 50.0),
        "min_dist_range": (60.0, 350.0),
        "distance_range": (50.0, 2500.0),
        "include_flutter": False,
        "snr_range": (3.0, 18.0),
        "gain_range": (0.8, 1.2),
    },
    "Ground_Combustion_Noise": {
        "doppler_prob": 0.0,
        "velocity_range": (0.0, 0.0),
        "min_dist_range": (50.0, 200.0),
        "distance_range": (10.0, 500.0),
        "include_flutter": False,
        "snr_range": (4.0, 18.0),
        "gain_range": (0.75, 1.25),
    },
    "Environmental_Background": {
        "doppler_prob": 0.0,
        "velocity_range": (0.0, 0.0),
        "min_dist_range": (50.0, 200.0),
        "distance_range": (10.0, 600.0),
        "include_flutter": False,
        "snr_range": (2.0, 18.0),
        "gain_range": (0.6, 1.3),
        "mix_bg_prob": 0.65,
    },
}


def _to_float32_mono(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim > 1:
        arr = np.mean(arr, axis=-1, dtype=np.float32)
    return np.ascontiguousarray(arr, dtype=np.float32)


def apply_atmospheric_absorption(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    distance_m: float = 300.0,
    alpha0: float = DEFAULT_ALPHA0,
) -> np.ndarray:
    """Apply ISO 9613-1 inspired frequency attenuation and distance low-pass rolloff.

    A(f, d) = exp(-alpha(f) * d / (20 * log10(e))), alpha(f) = alpha0 * f^2,
    combined with smooth low-pass cutoff f_c(d) = max(1500, 8000 / (1 + 0.0008 * d)).
    """
    x = _to_float32_mono(audio)
    if x.size == 0 or distance_m <= 0.0:
        return x

    d = float(distance_m)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / sr)
    spec = np.fft.rfft(x.astype(np.float64))

    twenty_log10_e = 20.0 * np.log10(np.e)
    alpha_f = alpha0 * (freqs ** 2)
    atten = np.exp(-alpha_f * d / twenty_log10_e)

    f_c = max(1500.0, 8000.0 / (1.0 + 0.0008 * d))
    lpf = 1.0 / np.sqrt(1.0 + (freqs / f_c) ** 4)

    filtered = np.fft.irfft(spec * atten * lpf, n=x.size)
    return filtered.astype(np.float32, copy=False)


def apply_doppler_shift(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    velocity_mps: float = 50.0,
    min_distance_m: float = 150.0,
    t_center: float = 0.5,
) -> np.ndarray:
    """Apply time-varying flyby Doppler shift via integrated phase trajectory resampling."""
    x = _to_float32_mono(audio)
    n = x.size
    if n < 4 or abs(velocity_mps) < 1e-6:
        return x

    v_s = float(velocity_mps)
    d_0 = max(1.0, float(min_distance_m))
    t_orig = np.arange(n, dtype=np.float64) / float(sr)
    dt = 1.0 / float(sr)

    rel_t = t_orig - float(t_center)
    radial_vel = v_s * (v_s * rel_t) / np.sqrt(d_0 ** 2 + (v_s * rel_t) ** 2)
    # Guard against supersonic singularities
    denom = np.maximum(SPEED_OF_SOUND_MPS + radial_vel, 10.0)
    freq_ratio = SPEED_OF_SOUND_MPS / denom

    tau = np.zeros(n, dtype=np.float64)
    tau[1:] = np.cumsum(0.5 * (freq_ratio[:-1] + freq_ratio[1:])) * dt

    interpolator = interp1d(
        t_orig,
        x.astype(np.float64),
        kind="linear",
        bounds_error=False,
        fill_value=(float(x[0]), float(x[-1])),
    )
    warped = interpolator(tau)
    return warped.astype(np.float32, copy=False)


def apply_blade_flutter(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    f_b: float | None = None,
    depth: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply rotor/propeller blade-pass amplitude modulation y(t) = x(t)*(1 + m*sin(2*pi*f_b*t))."""
    x = _to_float32_mono(audio)
    if x.size == 0:
        return x

    gen = rng if rng is not None else np.random.default_rng()
    freq_hz = float(f_b) if f_b is not None else float(gen.uniform(15.0, 40.0))
    mod_depth = float(depth) if depth is not None else float(gen.uniform(0.05, 0.15))

    t = np.arange(x.size, dtype=np.float32) / float(sr)
    envelope = 1.0 + mod_depth * np.sin(2.0 * np.pi * freq_hz * t)
    return (x * envelope).astype(np.float32, copy=False)


def mix_background_noise(
    target: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    """Mix target waveform with background noise at a specified SNR in dB."""
    s = _to_float32_mono(target)
    n = _to_float32_mono(noise)
    if s.size == 0:
        return s
    if n.size == 0:
        return s

    if n.size < s.size:
        repeats = int(np.ceil(s.size / n.size))
        n = np.tile(n, repeats)[: s.size]
    elif n.size > s.size:
        n = n[: s.size]

    norm_s = float(np.linalg.norm(s))
    norm_n = float(np.linalg.norm(n))
    if norm_n < 1e-12:
        return s
    if norm_s < 1e-12:
        return (10.0 ** (-snr_db / 20.0)) * n

    scale = (10.0 ** (-float(snr_db) / 20.0)) * (norm_s / norm_n)
    mixed = s + scale * n
    return mixed.astype(np.float32, copy=False)


def augment_clip(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    rng: np.random.Generator | None = None,
    noise_pool: list[np.ndarray] | None = None,
    include_flutter: bool = True,
    class_name: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a randomized class-aware physics augmentation pipeline and return (clip, params)."""
    gen = rng if rng is not None else np.random.default_rng()
    out = _to_float32_mono(audio)

    profile = CLASS_AUGMENT_PROFILES.get(class_name) if class_name else None
    doppler_prob = float(profile["doppler_prob"]) if profile else 0.75
    vel_lo, vel_hi = profile["velocity_range"] if profile else (15.0, 55.0)
    dist0_lo, dist0_hi = profile["min_dist_range"] if profile else (80.0, 400.0)
    dist_lo, dist_hi = profile["distance_range"] if profile else (50.0, 4000.0)
    use_flutter = bool(profile["include_flutter"]) if profile else bool(include_flutter)
    snr_lo, snr_hi = profile["snr_range"] if profile else (-10.0, 15.0)
    gain_lo, gain_hi = profile["gain_range"] if profile else (1.0, 1.0)
    mix_bg_prob = float(profile.get("mix_bg_prob", 1.0)) if profile else 1.0

    velocity = 0.0
    min_dist = float(gen.uniform(dist0_lo, dist0_hi))
    if doppler_prob > 0.0 and vel_hi > 0.0 and gen.random() < doppler_prob:
        velocity = float(gen.uniform(vel_lo, vel_hi))
        t_center = float(gen.uniform(0.2, 0.8)) * (out.size / float(sr))
        out = apply_doppler_shift(
            out,
            sr=sr,
            velocity_mps=velocity,
            min_distance_m=min_dist,
            t_center=t_center,
        )

    distance_m = float(gen.uniform(dist_lo, dist_hi))
    out = apply_atmospheric_absorption(out, sr=sr, distance_m=distance_m)

    f_b = None
    depth = None
    if use_flutter and gen.random() < 0.8:
        f_b = float(gen.uniform(15.0, 40.0))
        depth = float(gen.uniform(0.05, 0.15))
        out = apply_blade_flutter(out, sr=sr, f_b=f_b, depth=depth, rng=gen)

    gain = float(gen.uniform(gain_lo, gain_hi)) if gain_hi > gain_lo else float(gain_lo)
    if abs(gain - 1.0) > 1e-4:
        out = (out * gain).astype(np.float32, copy=False)

    snr_db: float | None = None
    if gen.random() < mix_bg_prob:
        snr_db = float(gen.uniform(snr_lo, snr_hi))
        if noise_pool:
            bg = noise_pool[int(gen.integers(0, len(noise_pool)))]
        else:
            bg = gen.normal(0.0, 0.02, size=out.size).astype(np.float32)
        out = mix_background_noise(out, bg, snr_db=snr_db)

    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.99:
        out = (0.95 * out / peak).astype(np.float32)

    params = {
        "class_name": class_name,
        "distance_m": round(distance_m, 2),
        "velocity_mps": round(velocity, 2),
        "min_distance_m": round(min_dist, 2),
        "flutter_hz": round(f_b, 2) if f_b is not None else None,
        "flutter_depth": round(depth, 4) if depth is not None else None,
        "gain": round(gain, 4),
        "snr_db": round(snr_db, 2) if snr_db is not None else None,
    }
    return out, params


def _detect_class_from_rel_path(rel_parent: Path) -> str | None:
    if rel_parent.parts and rel_parent.parts[0] in CLASSES:
        return rel_parent.parts[0]
    return None


def batch_augment_directory(
    input_dir: Path | str,
    output_dir: Path | str,
    background_dir: Path | str | None = None,
    augmentations_per_file: int = 4,
    target_per_class: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Batch-augment clean WAV files into output_dir with class-aware physics perturbations."""
    in_root = Path(input_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    noise_pool: list[np.ndarray] = []
    if background_dir is not None:
        bg_root = Path(background_dir)
        for bg_wav in sorted(bg_root.rglob("*.wav")):
            lower_path = str(bg_wav).lower()
            if "cessna" in lower_path or "helicopter" in lower_path:
                continue
            bg_audio, _ = sf.read(bg_wav, dtype="float32")
            noise_pool.append(_to_float32_mono(bg_audio))

    wav_files = sorted(in_root.rglob("*.wav"))
    class_file_counts: dict[str | None, int] = {}
    for wav_path in wav_files:
        cls_name = _detect_class_from_rel_path(wav_path.relative_to(in_root).parent)
        class_file_counts[cls_name] = class_file_counts.get(cls_name, 0) + 1

    records: list[dict[str, Any]] = []
    per_class_counts: dict[str, int] = {}

    for wav_path in wav_files:
        audio, orig_sr = sf.read(wav_path, dtype="float32")
        mono = _to_float32_mono(audio)
        sr = int(orig_sr) if orig_sr else SAMPLE_RATE

        rel_parent = wav_path.relative_to(in_root).parent
        target_dir = out_root / rel_parent
        target_dir.mkdir(parents=True, exist_ok=True)

        cls_name = _detect_class_from_rel_path(rel_parent)
        if target_per_class is not None and target_per_class > 0:
            cls_sources = max(1, class_file_counts.get(cls_name, 1))
            n_augs = max(1, min(6, int(round(target_per_class / cls_sources))))
        else:
            n_augs = int(augmentations_per_file)

        for aug_idx in range(n_augs):
            augmented, params = augment_clip(
                mono,
                sr=sr,
                rng=rng,
                noise_pool=noise_pool if noise_pool else None,
                class_name=cls_name,
            )
            out_path = target_dir / f"{wav_path.stem}_aug_{aug_idx:02d}.wav"
            sf.write(out_path, np.clip(augmented, -1.0, 1.0), sr, subtype="PCM_16")
            cls_key = cls_name or "unclassified"
            per_class_counts[cls_key] = per_class_counts.get(cls_key, 0) + 1
            records.append(
                {
                    "source": str(wav_path),
                    "output": str(out_path),
                    "params": params,
                }
            )

    manifest = {
        "input_dir": str(in_root),
        "output_dir": str(out_root),
        "source_count": len(wav_files),
        "augmented_count": len(records),
        "target_per_class": target_per_class,
        "per_class_augmented_counts": per_class_counts,
        "records": records,
    }
    (out_root / "augment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply class-aware physics-based acoustic augmentations to WAV corpus."
    )
    parser.add_argument("--input-dir", type=Path, default=CLEAN_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=AUGMENTED_DATA_DIR)
    parser.add_argument("--background-dir", type=Path, default=None)
    parser.add_argument("--augmentations-per-file", type=int, default=4)
    parser.add_argument(
        "--target-per-class",
        type=int,
        default=None,
        help="Optional target augmented clips per class (caps per-file factor between 1 and 6).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = batch_augment_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        background_dir=args.background_dir,
        augmentations_per_file=args.augmentations_per_file,
        target_per_class=args.target_per_class,
        seed=args.seed,
    )
    print(
        f"Augmented {manifest['source_count']} files -> "
        f"{manifest['augmented_count']} clips in {args.output_dir} "
        f"(per_class={manifest['per_class_augmented_counts']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
