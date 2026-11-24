"""Audio cleaning, vocal isolation, siren/gunfire rejection, and harmonic comb verification."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal

from config import (
    CLASSES,
    CLASS_CLEANING_PROFILES,
    CLEAN_DATA_DIR,
    EPS,
    NUM_SAMPLES,
    RAW_DATA_DIR,
    SAMPLE_RATE,
    SIREN_GUNFIRE_LABELS,
    SIREN_GUNFIRE_THRESHOLD,
)


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim > 1:
        arr = np.mean(arr, axis=-1, dtype=np.float32)
    return np.ascontiguousarray(arr, dtype=np.float32)


def _resample_if_needed(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr or audio.size == 0:
        return audio
    gcd = int(np.gcd(orig_sr, target_sr))
    up = target_sr // gcd
    down = orig_sr // gcd
    return signal.resample_poly(audio, up, down).astype(np.float32, copy=False)


def _moving_median_1d(values: np.ndarray, kernel_size: int) -> np.ndarray:
    """Compute odd-window 1D moving median for smooth spectral envelope estimation."""
    k = max(3, kernel_size | 1)
    pad = k // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, k)
    return np.median(windows, axis=-1).astype(values.dtype, copy=False)


def _spectral_speech_attenuation(audio: np.ndarray, sr: int) -> np.ndarray:
    """Fallback speech-band attenuator preserving narrow harmonic comb peaks.

    In the 300-3400 Hz speech formant range, broad spectral envelopes (e.g. spoken
    vowels and conversational chatter) are suppressed while sharp harmonic peaks
    characteristic of UAV rotors/engines are preserved.
    """
    x = _to_mono_float32(audio)
    if x.size < 16:
        return x

    n_fft = min(1024, 1 << int(np.floor(np.log2(max(32, x.size)))))
    hop = max(16, n_fft // 4)
    freqs, _, stft = signal.stft(
        x,
        fs=sr,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop,
        nfft=n_fft,
        boundary="zeros",
        padded=True,
    )
    mag = np.abs(stft)
    speech_mask = (freqs >= 300.0) & (freqs <= 3400.0)

    if np.any(speech_mask):
        gain = np.ones_like(mag, dtype=np.float32)
        speech_mag = mag[speech_mask, :]
        # Estimate smooth formant envelope across frequency bins per time frame
        for t in range(speech_mag.shape[1]):
            col = speech_mag[:, t]
            smooth_env = _moving_median_1d(col, kernel_size=11)
            # Suppress energy close to the broad formant floor while passing sharp harmonics
            excess = np.maximum(col - 0.85 * smooth_env, 0.0)
            frame_gain = np.maximum(0.15, excess / (col + 1e-8))
            gain[speech_mask, t] = frame_gain
        stft = stft * gain

    _, reconstructed = signal.istft(
        stft,
        fs=sr,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop,
        nfft=n_fft,
        input_onesided=True,
        boundary=True,
    )
    if reconstructed.size >= x.size:
        out = reconstructed[: x.size]
    else:
        out = np.pad(reconstructed, (0, x.size - reconstructed.size))
    return out.astype(np.float32, copy=False)


def _demucs_isolate_other_stem(audio: np.ndarray, sr: int) -> np.ndarray:
    """Isolate Demucs htdemucs 'other' stem (removing vocals/drums/bass)."""
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    x = _to_mono_float32(audio)
    model = get_model("htdemucs")
    model.eval()
    model_sr = int(model.samplerate)
    resampled = _resample_if_needed(x, sr, model_sr)

    # Demucs expects [batch, channels, samples]
    stereo = np.stack([resampled, resampled], axis=0)
    wav_tensor = torch.from_numpy(stereo).unsqueeze(0).to(torch.float32)

    with torch.no_grad():
        sources = apply_model(model, wav_tensor, split=True, overlap=0.25, progress=False)

    other_idx = model.sources.index("other") if "other" in model.sources else -1
    other_stereo = sources[0, other_idx].cpu().numpy()
    other_mono = np.mean(other_stereo, axis=0, dtype=np.float32)
    restored = _resample_if_needed(other_mono, model_sr, sr)
    if restored.size >= x.size:
        return restored[: x.size].astype(np.float32, copy=False)
    return np.pad(restored, (0, x.size - restored.size)).astype(np.float32, copy=False)


def isolate_non_vocal(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    backend: str = "auto",
) -> np.ndarray:
    """Isolate non-vocal acoustic content using Demucs ('other' stem) or spectral fallback.

    Args:
        audio: 1D (or multi-channel) audio waveform.
        sr: Sample rate in Hz.
        backend: "auto" (use demucs if available else spectral fallback),
                 "demucs" (require demucs), or "spectral" (always spectral fallback).
    """
    if backend not in {"auto", "demucs", "spectral"}:
        raise ValueError(f"Unsupported backend '{backend}'; expected auto|demucs|spectral.")

    if backend == "spectral":
        return _spectral_speech_attenuation(audio, sr)

    if backend == "demucs":
        return _demucs_isolate_other_stem(audio, sr)

    try:
        import demucs  # noqa: F401
        return _demucs_isolate_other_stem(audio, sr)
    except Exception:
        return _spectral_speech_attenuation(audio, sr)


class SirenGunfireFilter:
    """Reject 1-second frames containing sirens, fire alarms, or gunshots/impulses."""

    REJECT_LABELS: set[str] = set(SIREN_GUNFIRE_LABELS)

    def __init__(
        self,
        threshold: float = SIREN_GUNFIRE_THRESHOLD,
        scorer: Callable[[np.ndarray, int], dict[str, float]] | None = None,
        backend: str = "auto",
    ) -> None:
        if backend not in {"auto", "yamnet", "heuristic"}:
            raise ValueError(f"Unsupported backend '{backend}'.")
        self.threshold = float(threshold)
        self._custom_scorer = scorer
        self._backend = backend
        self._yamnet_model: Any = None
        self._yamnet_class_names: list[str] | None = None

        if self._custom_scorer is None and backend in {"auto", "yamnet"}:
            try:
                self._load_yamnet()
            except Exception:
                if backend == "yamnet":
                    raise
                self._backend = "heuristic"

    def _load_yamnet(self) -> None:
        import tensorflow_hub as hub

        model = hub.load("https://tfhub.dev/google/yamnet/1")
        class_map_path = model.class_map_path().numpy().decode("utf-8")
        names: list[str] = []
        with open(class_map_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                names.append(row["display_name"])
        self._yamnet_model = model
        self._yamnet_class_names = names
        self._backend = "yamnet"

    @property
    def backend_name(self) -> str:
        if self._custom_scorer is not None:
            return "custom"
        return self._backend

    def _yamnet_score(self, frame: np.ndarray, sr: int) -> dict[str, float]:
        import tensorflow as tf

        wav = _resample_if_needed(_to_mono_float32(frame), sr, 16000)
        scores, _, _ = self._yamnet_model(tf.convert_to_tensor(wav, dtype=tf.float32))
        mean_scores = scores.numpy().mean(axis=0)
        names = self._yamnet_class_names or []
        return {
            label: float(mean_scores[names.index(label)])
            for label in self.REJECT_LABELS
            if label in names
        }

    @staticmethod
    def _heuristic_score(frame: np.ndarray, sr: int) -> dict[str, float]:
        """Lightweight spectral siren + transient impulse heuristic fallback."""
        x = _to_mono_float32(frame)
        if x.size < 64:
            return {label: 0.0 for label in SIREN_GUNFIRE_LABELS}

        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        if rms < 1e-5:
            return {label: 0.0 for label in SIREN_GUNFIRE_LABELS}

        # 1. Impulse / gunshot detector via crest factor + 10ms chunk burst ratio
        chunk_len = max(16, int(round(0.010 * sr)))
        n_chunks = max(1, x.size // chunk_len)
        trimmed = x[: n_chunks * chunk_len].reshape(n_chunks, chunk_len)
        chunk_rms = np.sqrt(np.mean(trimmed * trimmed, axis=1) + 1e-12)
        median_chunk = float(np.median(chunk_rms))
        peak_chunk = float(np.max(chunk_rms))
        burst_ratio = peak_chunk / (median_chunk + 1e-8)
        crest = float(np.max(np.abs(x))) / (rms + 1e-8)

        gunshot_score = 0.0
        if burst_ratio > 3.5 and crest > 6.0:
            gunshot_score = float(
                np.clip(
                    0.35 * ((burst_ratio - 3.5) / 4.0) + 0.65 * ((crest - 6.0) / 8.0),
                    0.0,
                    1.0,
                )
            )
            # Boost clear sharp transients where peak energy is concentrated in <= 15% of clip
            active_frac = float(np.mean(chunk_rms > 0.25 * peak_chunk))
            if active_frac < 0.25:
                gunshot_score = float(np.clip(max(gunshot_score, 0.75 * (1.0 - active_frac)), 0.0, 1.0))

        # 2. Siren / fire alarm detector via short-time dominant tonal pitch + modulation
        n_fft = min(1024, max(256, 1 << int(np.floor(np.log2(max(256, int(0.05 * sr)))))))
        hop = max(64, n_fft // 2)
        freqs, _, zxx = signal.stft(
            x,
            fs=sr,
            window="hann",
            nperseg=n_fft,
            noverlap=n_fft - hop,
            nfft=n_fft,
            boundary=None,
        )
        power = np.abs(zxx) ** 2
        if power.shape[1] == 0:
            return {
                "Siren": 0.0,
                "Civil defense siren": 0.0,
                "Fire alarm": 0.0,
                "Gunshot, gunfire": gunshot_score,
            }

        low_mask = (freqs >= 50.0) & (freqs <= 350.0)
        siren_mask = (freqs >= 550.0) & (freqs <= 3800.0)
        low_energy = float(np.sum(power[low_mask, :]))
        siren_energy = float(np.sum(power[siren_mask, :]))
        total_energy = float(np.sum(power)) + 1e-12

        siren_band_power = power[siren_mask, :]
        siren_freqs = freqs[siren_mask]
        frame_totals = np.sum(power, axis=0) + 1e-12

        # Peak concentration within +-1 bin around dominant peak per frame
        peak_bins = np.argmax(siren_band_power, axis=0)
        peak_energy = np.zeros(siren_band_power.shape[1], dtype=np.float32)
        for t_idx, p_bin in enumerate(peak_bins):
            lo = max(0, p_bin - 1)
            hi = min(siren_band_power.shape[0], p_bin + 2)
            peak_energy[t_idx] = np.sum(siren_band_power[lo:hi, t_idx])

        tonality = float(np.mean(peak_energy / frame_totals))
        peak_hz_trace = siren_freqs[peak_bins]
        freq_sweep_std = float(np.std(peak_hz_trace))
        low_ratio = low_energy / total_energy
        siren_ratio = siren_energy / total_energy

        # Amplitude modulation depth for pulsed fire alarms
        frame_env = np.sqrt(frame_totals)
        env_mod = float(np.std(frame_env) / (np.mean(frame_env) + 1e-8))

        siren_score = 0.0
        alarm_score = 0.0
        # Sirens/alarms dominate 550-3800 Hz with high single-tone concentration and minimal <350 Hz fundamental
        if siren_ratio > 0.65 and low_ratio < 0.15 and tonality > 0.45:
            if freq_sweep_std > 40.0:
                siren_score = float(
                    np.clip(0.45 + 0.35 * tonality + 0.20 * min(1.0, freq_sweep_std / 250.0), 0.0, 1.0)
                )
            elif env_mod > 0.35:
                alarm_score = float(
                    np.clip(0.45 + 0.35 * tonality + 0.20 * min(1.0, env_mod / 0.8), 0.0, 1.0)
                )

        return {
            "Siren": siren_score,
            "Civil defense siren": siren_score * 0.95,
            "Fire alarm": alarm_score,
            "Gunshot, gunfire": gunshot_score,
        }

    def score_frame(self, frame: np.ndarray, sr: int = SAMPLE_RATE) -> dict[str, float]:
        """Return class probability scores for siren/alarm/gunfire categories."""
        if self._custom_scorer is not None:
            raw = self._custom_scorer(frame, sr)
            return {str(k): float(v) for k, v in raw.items()}
        if self._backend == "yamnet" and self._yamnet_model is not None:
            return self._yamnet_score(frame, sr)
        return self._heuristic_score(frame, sr)

    def is_rejected(
        self,
        frame: np.ndarray,
        sr: int = SAMPLE_RATE,
    ) -> tuple[bool, dict[str, float]]:
        """Evaluate a 1-second frame and return (rejected, scores_dict)."""
        scores = self.score_frame(frame, sr)
        rejected = any(scores.get(label, 0.0) > self.threshold for label in self.REJECT_LABELS)
        return rejected, scores

    def should_reject(self, frame: np.ndarray, sr: int = SAMPLE_RATE) -> bool:
        """Return True when any target hazard label exceeds threshold."""
        rejected, _ = self.is_rejected(frame, sr)
        return rejected


def _normalized_autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute normalized autocorrelation R(tau) for tau in [0, max_lag]."""
    n = x.size
    centered = x.astype(np.float64) - float(np.mean(x))
    n_fft = 1 << int(np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(centered, n=n_fft)
    ac_raw = np.fft.irfft(spec * np.conj(spec), n=n_fft)[: max_lag + 1]

    sq = centered * centered
    csum = np.concatenate(([0.0], np.cumsum(sq)))
    taus = np.arange(max_lag + 1)
    # Energy of x[0 : n - tau] and x[tau : n]
    e_left = csum[n - taus]
    e_right = csum[n] - csum[taus]
    denom = np.sqrt(np.maximum(e_left * e_right, 0.0)) + 1e-12
    return (ac_raw / denom).astype(np.float64, copy=False)


def _first_main_lobe_end(r: np.ndarray) -> int:
    """Return index of first zero-crossing after lag 0, or first local minimum if always positive."""
    n = r.size
    for tau in range(1, n):
        if r[tau] <= 0.0:
            return tau
    for tau in range(1, n - 1):
        if r[tau] <= r[tau - 1] and r[tau] <= r[tau + 1]:
            return tau
    return 1


def _local_maxima_indices(r: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Return indices t in [lo, hi] that are strict local peaks r[t] > r[t-1] and r[t] >= r[t+1]."""
    start = max(1, int(lo))
    end = min(int(r.size) - 2, int(hi))
    if end < start:
        return np.empty(0, dtype=np.int64)
    center = r[start : end + 1]
    left = r[start - 1 : end]
    right = r[start + 1 : end + 2]
    rel = np.where((center > left) & (center >= right))[0]
    return (start + rel).astype(np.int64, copy=False)


def compute_harmonic_comb_metrics(
    frame: np.ndarray,
    sr: int = SAMPLE_RATE,
    min_hz: float = 50.0,
    max_hz: float = 300.0,
) -> tuple[float, float]:
    """Return (estimated_f0_hz, peak_normalized_autocorrelation) inside [min_hz, max_hz]."""
    x = _to_mono_float32(frame)
    tau_min = max(1, int(np.floor(sr / max_hz)))
    tau_max = int(np.ceil(sr / min_hz))
    if x.size <= tau_max + 8 or float(np.std(x)) < 1e-6:
        return 0.0, 0.0

    search_max = min(x.size - 2, max(tau_max + 2, int(np.ceil(sr / 25.0))))
    r = _normalized_autocorrelation(x, search_max)

    z0 = _first_main_lobe_end(r)
    target_lo = max(tau_min, z0)
    if target_lo > tau_max:
        return 0.0, 0.0

    target_peaks = _local_maxima_indices(r, target_lo, tau_max)
    if target_peaks.size > 0:
        best_pos = int(np.argmax(r[target_peaks]))
        peak_tau = int(target_peaks[best_pos])
        peak_corr = float(r[peak_tau])
    else:
        band = r[target_lo : tau_max + 1]
        if band.size == 0:
            return 0.0, 0.0
        rel_peak = int(np.argmax(band))
        peak_tau = target_lo + rel_peak
        peak_corr = float(band[rel_peak])

    if peak_corr <= 0.0:
        return 0.0, peak_corr

    # Reject signals whose true periodic peak lies above max_hz (only inspecting true local
    # maxima strictly past the zero-lag main lobe z0 so low-passed piston signals aren't clipped).
    fast_lo = max(z0, int(np.floor(sr / 2000.0)))
    fast_hi = min(tau_min - 1, int(np.floor(0.92 * tau_min)))
    fast_peaks = _local_maxima_indices(r, fast_lo, fast_hi)
    if fast_peaks.size > 0:
        best_fast = int(fast_peaks[int(np.argmax(r[fast_peaks]))])
        if float(r[best_fast]) >= max(0.85 * peak_corr, 0.25):
            return float(sr / best_fast), 0.0

    # Reject signals whose true periodic peak lies below min_hz (lags longer than tau_max)
    slow_peaks = _local_maxima_indices(r, max(z0, tau_max + 1), search_max - 1)
    if slow_peaks.size > 0:
        best_slow = int(slow_peaks[int(np.argmax(r[slow_peaks]))])
        if float(r[best_slow]) > 1.05 * peak_corr:
            return float(sr / best_slow), 0.0

    return float(sr / peak_tau), peak_corr


def verify_harmonic_comb(
    frame: np.ndarray,
    sr: int = SAMPLE_RATE,
    min_hz: float = 50.0,
    max_hz: float = 300.0,
    threshold: float = 0.25,
) -> bool:
    """Verify periodic harmonic comb energy in [min_hz, max_hz] via autocorrelation."""
    f0_hz, peak_corr = compute_harmonic_comb_metrics(
        frame=frame, sr=sr, min_hz=min_hz, max_hz=max_hz
    )
    return bool(min_hz <= f0_hz <= max_hz and peak_corr >= threshold)


def slice_into_one_second_frames(audio: np.ndarray, sr: int = SAMPLE_RATE) -> list[np.ndarray]:
    """Split a waveform into 1-second (sr-sample) frames, padding short tails >= 0.5s."""
    x = _to_mono_float32(audio)
    frame_len = int(sr)
    if x.size == 0:
        return []
    if x.size < frame_len:
        return [np.pad(x, (0, frame_len - x.size)).astype(np.float32)]

    frames: list[np.ndarray] = []
    for start in range(0, x.size, frame_len):
        chunk = x[start : start + frame_len]
        if chunk.size == frame_len:
            frames.append(chunk)
        elif chunk.size >= frame_len // 2:
            frames.append(np.pad(chunk, (0, frame_len - chunk.size)).astype(np.float32))
    return frames


def clean_directory(
    input_dir: Path | str,
    output_dir: Path | str,
    vocal_backend: str = "auto",
    siren_threshold: float = SIREN_GUNFIRE_THRESHOLD,
    comb_threshold: float = 0.25,
    comb_min_hz: float = 50.0,
    comb_max_hz: float = 300.0,
    require_comb: bool = True,
    apply_siren_filter: bool = True,
) -> dict[str, Any]:
    """Clean all WAV files under input_dir and export validated 1s WAV clips to output_dir."""
    in_root = Path(input_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    siren_filter = SirenGunfireFilter(threshold=siren_threshold) if apply_siren_filter else None
    wav_files = sorted(in_root.rglob("*.wav"))

    total_frames = 0
    kept_frames = 0
    rejected_siren = 0
    rejected_comb = 0
    kept_paths: list[str] = []

    for wav_path in wav_files:
        audio, orig_sr = sf.read(wav_path, dtype="float32")
        mono = _resample_if_needed(_to_mono_float32(audio), int(orig_sr), SAMPLE_RATE)
        non_vocal = isolate_non_vocal(mono, sr=SAMPLE_RATE, backend=vocal_backend)
        frames = slice_into_one_second_frames(non_vocal, sr=SAMPLE_RATE)

        rel_parent = wav_path.relative_to(in_root).parent
        target_dir = out_root / rel_parent
        target_dir.mkdir(parents=True, exist_ok=True)

        for idx, frame in enumerate(frames):
            total_frames += 1
            if siren_filter is not None and siren_filter.should_reject(frame, sr=SAMPLE_RATE):
                rejected_siren += 1
                continue
            if require_comb and not verify_harmonic_comb(
                frame,
                sr=SAMPLE_RATE,
                min_hz=comb_min_hz,
                max_hz=comb_max_hz,
                threshold=comb_threshold,
            ):
                rejected_comb += 1
                continue

            out_wav = target_dir / f"{wav_path.stem}_frame_{idx:04d}.wav"
            clipped = np.clip(frame, -1.0, 1.0)
            sf.write(out_wav, clipped, SAMPLE_RATE, subtype="PCM_16")
            kept_frames += 1
            kept_paths.append(str(out_wav))

    summary = {
        "input_dir": str(in_root),
        "output_dir": str(out_root),
        "source_files": len(wav_files),
        "total_frames": total_frames,
        "kept_frames": kept_frames,
        "rejected_siren_gunfire": rejected_siren,
        "rejected_harmonic_comb": rejected_comb,
        "comb_min_hz": comb_min_hz,
        "comb_max_hz": comb_max_hz,
        "apply_siren_filter": apply_siren_filter,
        "siren_backend": siren_filter.backend_name if siren_filter is not None else "disabled",
        "kept_files": kept_paths,
    }
    (out_root / "clean_report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean and filter 16kHz UAV audio (vocal isolation, siren/gunfire filter, comb verify)."
    )
    parser.add_argument("--input-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=CLEAN_DATA_DIR)
    parser.add_argument(
        "--class-profile",
        choices=list(CLASSES),
        default=None,
        help="Optional class cleaning profile from config.CLASS_CLEANING_PROFILES.",
    )
    parser.add_argument(
        "--vocal-backend",
        choices=["auto", "demucs", "spectral"],
        default="auto",
    )
    parser.add_argument("--siren-threshold", type=float, default=SIREN_GUNFIRE_THRESHOLD)
    parser.add_argument("--comb-threshold", type=float, default=None)
    parser.add_argument("--comb-min-hz", type=float, default=None)
    parser.add_argument("--comb-max-hz", type=float, default=None)
    parser.add_argument(
        "--skip-comb-check",
        action="store_true",
        help="Disable harmonic comb filtering (useful for pure background corpora).",
    )
    parser.add_argument(
        "--skip-siren-filter",
        action="store_true",
        help="Disable siren/gunfire rejection (useful when preserving background sirens).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    profile = CLASS_CLEANING_PROFILES[args.class_profile] if args.class_profile else {}

    comb_min_hz = args.comb_min_hz if args.comb_min_hz is not None else float(profile.get("comb_min_hz", 50.0))
    comb_max_hz = args.comb_max_hz if args.comb_max_hz is not None else float(profile.get("comb_max_hz", 300.0))
    comb_threshold = args.comb_threshold if args.comb_threshold is not None else float(profile.get("comb_threshold", 0.25))
    require_comb = False if args.skip_comb_check else bool(profile.get("require_comb", True))
    apply_siren_filter = False if args.skip_siren_filter else bool(profile.get("apply_siren_filter", True))

    summary = clean_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        vocal_backend=args.vocal_backend,
        siren_threshold=args.siren_threshold,
        comb_threshold=comb_threshold,
        comb_min_hz=comb_min_hz,
        comb_max_hz=comb_max_hz,
        require_comb=require_comb,
        apply_siren_filter=apply_siren_filter,
    )
    print(
        f"Cleaned {summary['source_files']} files: kept {summary['kept_frames']}/{summary['total_frames']} "
        f"frames (rejected siren/gunfire={summary['rejected_siren_gunfire']}, "
        f"comb={summary['rejected_harmonic_comb']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
