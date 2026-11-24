"""Physically grounded synthetic acoustic corpus generator for all 5 UAV/noise classes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from config import CLASSES, CLASS_TO_IDX, DATA_DIR, NUM_SAMPLES, SAMPLE_RATE
from physics_augment import (
    apply_atmospheric_absorption,
    apply_blade_flutter,
    apply_doppler_shift,
    mix_background_noise,
)


def _pink_noise(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Generate 1/f pink noise normalized to unit RMS."""
    white = rng.normal(0.0, 1.0, size=n_samples)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / float(SAMPLE_RATE))
    scale = np.ones_like(freqs)
    nz = freqs > 0.0
    scale[nz] = 1.0 / np.sqrt(freqs[nz])
    pink = np.fft.irfft(spec * scale, n=n_samples)
    rms = float(np.sqrt(np.mean(pink * pink) + 1e-12))
    return (pink / rms).astype(np.float32, copy=False)


def _brown_noise(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Generate 1/f^2 wind/rumble brown noise normalized to unit RMS."""
    white = rng.normal(0.0, 1.0, size=n_samples)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / float(SAMPLE_RATE))
    scale = np.zeros_like(freqs)
    nz = freqs >= 20.0
    scale[nz] = 1.0 / freqs[nz]
    brown = np.fft.irfft(spec * scale, n=n_samples)
    rms = float(np.sqrt(np.mean(brown * brown) + 1e-12))
    return (brown / rms).astype(np.float32, copy=False)


def _synthesize_harmonic_comb(
    f0_inst: np.ndarray,
    sr: int,
    max_freq_hz: float,
    rolloff_power: float,
    rng: np.random.Generator,
    even_odd_bias: float = 0.0,
    pulse_sharpness: float = 0.0,
) -> np.ndarray:
    """Synthesize a band-limited harmonic comb from instantaneous fundamental f0_inst."""
    n = f0_inst.size
    phase0 = 2.0 * np.pi * np.cumsum(f0_inst.astype(np.float64)) / float(sr)
    phase0 += rng.uniform(0.0, 2.0 * np.pi)

    f0_mean = float(np.mean(f0_inst))
    max_harmonic = max(1, min(60, int(max_freq_hz / max(f0_mean, 20.0))))
    sig = np.zeros(n, dtype=np.float64)

    for h in range(1, max_harmonic + 1):
        if h * f0_mean >= 0.48 * sr:
            break
        amp = 1.0 / (float(h) ** rolloff_power)
        if even_odd_bias != 0.0:
            amp *= (1.0 + even_odd_bias) if (h % 2 == 1) else max(0.25, 1.0 - even_odd_bias)
        if pulse_sharpness > 0.0:
            amp *= np.exp(-float(h) * 0.02 * (1.0 - pulse_sharpness))
        phase_offset = rng.uniform(-0.2, 0.2) if h > 1 else 0.0
        sig += amp * np.sin(float(h) * phase0 + phase_offset)

    rms = float(np.sqrt(np.mean(sig * sig) + 1e-12))
    return (sig / rms).astype(np.float32, copy=False)


def _generate_shahed_piston(rng: np.random.Generator, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Shahed-136 / Geran-2 (MD-550 / Limbach L550E) two-stroke piston UAV sound."""
    n = sr
    t = np.arange(n, dtype=np.float64) / float(sr)
    regime = rng.random()

    # Cover 70-115 Hz firing rates, with dense emphasis on 75-100 Hz (including ~92.5 Hz)
    if regime < 0.40:
        # Steady / low-drift cruise (matches live browser simulation & steady cruise audio)
        f0 = float(rng.uniform(75.0, 102.0))
        drift = float(rng.uniform(-0.8, 0.8)) * np.sin(2.0 * np.pi * float(rng.uniform(0.3, 1.5)) * t)
        f0_inst = f0 + drift
        sig = _synthesize_harmonic_comb(
            f0_inst=f0_inst,
            sr=sr,
            max_freq_hz=4000.0,
            rolloff_power=float(rng.uniform(0.65, 1.0)),
            rng=rng,
            even_odd_bias=float(rng.uniform(0.0, 0.25)),
            pulse_sharpness=float(rng.uniform(0.2, 0.7)),
        )
        # Optional subtle piston pulse shaping
        if rng.random() < 0.5:
            sig = np.tanh(float(rng.uniform(1.1, 1.8)) * sig).astype(np.float32)

        sig = apply_atmospheric_absorption(sig, sr=sr, distance_m=float(rng.uniform(50.0, 650.0)))
        if rng.random() < 0.35:
            sig = apply_blade_flutter(
                sig,
                sr=sr,
                f_b=float(rng.uniform(15.0, 30.0)),
                depth=float(rng.uniform(0.03, 0.08)),
                rng=rng,
            )
        noise = 0.6 * _pink_noise(n, rng) + 0.4 * rng.normal(0.0, 1.0, n).astype(np.float32)
        sig = mix_background_noise(sig, noise, snr_db=float(rng.uniform(7.0, 26.0)))
    elif regime < 0.75:
        # Mid-range flyby with Doppler shift + atmospheric absorption
        f0 = float(rng.uniform(72.0, 112.0))
        f0_inst = np.full(n, f0, dtype=np.float64) + float(rng.uniform(-1.5, 1.5)) * np.sin(2.0 * np.pi * 0.8 * t)
        sig = _synthesize_harmonic_comb(
            f0_inst=f0_inst,
            sr=sr,
            max_freq_hz=4000.0,
            rolloff_power=float(rng.uniform(0.7, 1.05)),
            rng=rng,
            even_odd_bias=float(rng.uniform(-0.15, 0.25)),
            pulse_sharpness=float(rng.uniform(0.1, 0.6)),
        )
        sig = apply_doppler_shift(
            sig,
            sr=sr,
            velocity_mps=float(rng.uniform(35.0, 55.0)),
            min_distance_m=float(rng.uniform(90.0, 320.0)),
            t_center=float(rng.uniform(0.25, 0.75)),
        )
        sig = apply_atmospheric_absorption(sig, sr=sr, distance_m=float(rng.uniform(250.0, 1600.0)))
        if rng.random() < 0.5:
            sig = apply_blade_flutter(
                sig,
                sr=sr,
                f_b=float(rng.uniform(15.0, 32.0)),
                depth=float(rng.uniform(0.05, 0.11)),
                rng=rng,
            )
        noise = 0.7 * _pink_noise(n, rng) + 0.3 * _brown_noise(n, rng)
        sig = mix_background_noise(sig, noise, snr_db=float(rng.uniform(0.0, 16.0)))
    else:
        # Distant / degraded low-SNR Shahed acoustic signature
        f0 = float(rng.uniform(75.0, 108.0))
        f0_inst = np.full(n, f0, dtype=np.float64)
        sig = _synthesize_harmonic_comb(
            f0_inst=f0_inst,
            sr=sr,
            max_freq_hz=3800.0,
            rolloff_power=float(rng.uniform(0.75, 1.1)),
            rng=rng,
            even_odd_bias=float(rng.uniform(0.0, 0.2)),
        )
        sig = apply_doppler_shift(
            sig,
            sr=sr,
            velocity_mps=float(rng.uniform(25.0, 52.0)),
            min_distance_m=float(rng.uniform(150.0, 500.0)),
            t_center=float(rng.uniform(0.15, 0.85)),
        )
        sig = apply_atmospheric_absorption(sig, sr=sr, distance_m=float(rng.uniform(1000.0, 3400.0)))
        noise = 0.5 * _pink_noise(n, rng) + 0.5 * _brown_noise(n, rng)
        sig = mix_background_noise(sig, noise, snr_db=float(rng.uniform(-2.0, 8.0)))

    return np.clip(sig * float(rng.uniform(0.3, 0.85)), -1.0, 1.0).astype(np.float32)


def _generate_multirotor(rng: np.random.Generator, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Multirotor UAV (e.g. DJI Mavic): 130-260 Hz multi-rotor harmonics + 15-40 Hz blade flutter."""
    n = sr
    t = np.arange(n, dtype=np.float64) / float(sr)

    base_f0 = float(rng.uniform(135.0, 255.0))
    n_rotors = int(rng.integers(2, 5))
    sig = np.zeros(n, dtype=np.float32)

    for _ in range(n_rotors):
        rotor_f0 = base_f0 * float(rng.uniform(0.95, 1.05))
        # Subtle flight controller RPM stabilization jitter
        wobble_freq = float(rng.uniform(1.5, 5.5))
        wobble_amp = float(rng.uniform(1.0, 5.0))
        f0_inst = rotor_f0 + wobble_amp * np.sin(2.0 * np.pi * wobble_freq * t + rng.uniform(0, 6.28))
        rotor_sig = _synthesize_harmonic_comb(
            f0_inst=f0_inst,
            sr=sr,
            max_freq_hz=4000.0,
            rolloff_power=float(rng.uniform(0.85, 1.3)),
            rng=rng,
            even_odd_bias=float(rng.uniform(-0.2, 0.2)),
        )
        sig += rotor_sig / float(n_rotors)

    # Strong blade-pass flutter in [15, 40] Hz
    sig = apply_blade_flutter(
        sig,
        sr=sr,
        f_b=float(rng.uniform(15.0, 40.0)),
        depth=float(rng.uniform(0.06, 0.15)),
        rng=rng,
    )
    if rng.random() < 0.55:
        sig = apply_doppler_shift(
            sig,
            sr=sr,
            velocity_mps=float(rng.uniform(8.0, 28.0)),
            min_distance_m=float(rng.uniform(40.0, 220.0)),
            t_center=float(rng.uniform(0.2, 0.8)),
        )
    sig = apply_atmospheric_absorption(sig, sr=sr, distance_m=float(rng.uniform(50.0, 1200.0)))
    noise = 0.7 * _pink_noise(n, rng) + 0.3 * rng.normal(0.0, 1.0, n).astype(np.float32)
    sig = mix_background_noise(sig, noise, snr_db=float(rng.uniform(-3.0, 22.0)))
    return np.clip(sig * float(rng.uniform(0.3, 0.85)), -1.0, 1.0).astype(np.float32)


def _generate_fixed_wing_electric(rng: np.random.Generator, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Fixed-wing electric UAV: 250-450 Hz prop fundamental + high-frequency ESC whine."""
    n = sr
    t = np.arange(n, dtype=np.float64) / float(sr)

    f0 = float(rng.uniform(255.0, 445.0))
    f0_inst = f0 + float(rng.uniform(-3.0, 3.0)) * np.sin(2.0 * np.pi * 0.7 * t)
    prop_sig = _synthesize_harmonic_comb(
        f0_inst=f0_inst,
        sr=sr,
        max_freq_hz=3800.0,
        rolloff_power=float(rng.uniform(1.0, 1.5)),
        rng=rng,
    )

    # Characteristic brushless motor / ESC carrier whine in 1.8 - 3.8 kHz
    f_esc = float(rng.uniform(1800.0, 3800.0))
    esc_phase = 2.0 * np.pi * f_esc * t + rng.uniform(0.0, 6.28)
    esc_whine = (
        0.35 * np.sin(esc_phase)
        + 0.12 * np.sin(2.0 * np.pi * (f_esc * 0.5) * t + rng.uniform(0.0, 6.28))
    ).astype(np.float32)

    sig = (0.75 * prop_sig + 0.25 * esc_whine).astype(np.float32)
    sig = apply_blade_flutter(
        sig,
        sr=sr,
        f_b=float(rng.uniform(18.0, 40.0)),
        depth=float(rng.uniform(0.05, 0.12)),
        rng=rng,
    )
    if rng.random() < 0.7:
        sig = apply_doppler_shift(
            sig,
            sr=sr,
            velocity_mps=float(rng.uniform(18.0, 45.0)),
            min_distance_m=float(rng.uniform(70.0, 300.0)),
            t_center=float(rng.uniform(0.2, 0.8)),
        )
    sig = apply_atmospheric_absorption(sig, sr=sr, distance_m=float(rng.uniform(80.0, 2000.0)))
    noise = 0.6 * _pink_noise(n, rng) + 0.4 * rng.normal(0.0, 1.0, n).astype(np.float32)
    sig = mix_background_noise(sig, noise, snr_db=float(rng.uniform(0.0, 22.0)))
    return np.clip(sig * float(rng.uniform(0.3, 0.85)), -1.0, 1.0).astype(np.float32)


def _generate_ground_combustion(rng: np.random.Generator, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Ground combustion noise: rapid throttle revving, irregular idling, muffled low rumble."""
    n = sr
    t = np.arange(n, dtype=np.float64) / float(sr)
    mode = rng.random()

    if mode < 0.45:
        # Rapid revving / throttle blip (large non-stationary frequency sweep)
        f_start = float(rng.uniform(35.0, 75.0))
        f_peak = f_start * float(rng.uniform(1.6, 2.8))
        peak_t = float(rng.uniform(0.3, 0.7))
        width = float(rng.uniform(0.15, 0.35))
        f0_inst = f_start + (f_peak - f_start) * np.exp(-0.5 * ((t - peak_t) / width) ** 2)
        sig = _synthesize_harmonic_comb(
            f0_inst=f0_inst,
            sr=sr,
            max_freq_hz=2200.0,
            rolloff_power=float(rng.uniform(1.2, 1.9)),
            rng=rng,
        )
    elif mode < 0.80:
        # Heavy irregular diesel / 4-stroke idle (28-58 Hz + irregular firing jitter)
        f0 = float(rng.uniform(28.0, 58.0))
        jitter = rng.normal(0.0, float(rng.uniform(2.0, 6.0)), size=n)
        # Smooth jitter into realistic RPM hunting
        kernel = np.ones(400) / 400.0
        jitter = np.convolve(jitter, kernel, mode="same")
        f0_inst = np.maximum(20.0, f0 + jitter)
        sig = _synthesize_harmonic_comb(
            f0_inst=f0_inst,
            sr=sr,
            max_freq_hz=1500.0,
            rolloff_power=float(rng.uniform(1.4, 2.2)),
            rng=rng,
        )
        # Chugging 4-stroke amplitude modulation at 6-14 Hz (below UAV blade flutter range)
        chug = 1.0 + float(rng.uniform(0.25, 0.45)) * np.sin(2.0 * np.pi * float(rng.uniform(6.0, 14.0)) * t)
        sig = (sig * chug).astype(np.float32)
    else:
        # Distant highway / heavy truck exhaust + road tire rumble
        f0 = float(rng.uniform(40.0, 68.0))
        f0_inst = f0 + float(rng.uniform(-10.0, 15.0)) * t
        sig = _synthesize_harmonic_comb(
            f0_inst=f0_inst,
            sr=sr,
            max_freq_hz=1200.0,
            rolloff_power=float(rng.uniform(1.5, 2.3)),
            rng=rng,
        )

    # Ground muffler / terrain low-pass & heavy brown rumble
    sig = apply_atmospheric_absorption(sig, sr=sr, distance_m=float(rng.uniform(1200.0, 4000.0)))
    rumble = _brown_noise(n, rng)
    sig = mix_background_noise(sig, rumble, snr_db=float(rng.uniform(-4.0, 10.0)))
    return np.clip(sig * float(rng.uniform(0.3, 0.85)), -1.0, 1.0).astype(np.float32)


def _generate_environmental_background(rng: np.random.Generator, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Environmental background: wind, rain, sirens, impulses, speech-like bursts, silence."""
    n = sr
    t = np.arange(n, dtype=np.float64) / float(sr)
    choice = rng.random()

    if choice < 0.22:
        # Gusting wind (1/f^2 noise with slow amplitude swells)
        swell = 0.5 + 0.5 * np.sin(2.0 * np.pi * float(rng.uniform(0.4, 2.0)) * t + rng.uniform(0, 6.28))
        sig = (_brown_noise(n, rng) * swell).astype(np.float32)
    elif choice < 0.44:
        # Realistic outdoor wind + broadband hiss via 1-pole IIR leaky filters
        from scipy import signal as sp_signal

        white = rng.uniform(-1.0, 1.0, n).astype(np.float64)
        a_low = float(rng.uniform(0.92, 0.985))
        a_mid = float(rng.uniform(0.75, 0.90))
        low_wind = sp_signal.lfilter([1.0 - a_low], [1.0, -a_low], white)
        mid_pink = sp_signal.lfilter([1.0 - a_mid], [1.0, -a_mid], white)
        gust = 0.6 + 0.4 * np.sin(2.0 * np.pi * float(rng.uniform(0.2, 1.2)) * t + rng.uniform(0, 6.28))
        sig = (
            float(rng.uniform(0.08, 0.18)) * low_wind * gust
            + float(rng.uniform(0.02, 0.06)) * mid_pink
            + float(rng.uniform(0.008, 0.025)) * white
        ).astype(np.float32)
    elif choice < 0.62:
        # Rain / broadband white + pink hiss
        w = rng.normal(0.0, 1.0, n).astype(np.float32)
        p = _pink_noise(n, rng)
        sig = (0.55 * p + 0.45 * w).astype(np.float32)
    elif choice < 0.78:
        # Civil defense / emergency siren (FM tone sweep 600-1700 Hz)
        f_base = float(rng.uniform(600.0, 1000.0))
        f_span = float(rng.uniform(350.0, 800.0))
        mod_rate = float(rng.uniform(0.6, 2.5))
        f_inst = f_base + 0.5 * f_span * (1.0 + np.sin(2.0 * np.pi * mod_rate * t + rng.uniform(0, 6.28)))
        phase = 2.0 * np.pi * np.cumsum(f_inst) / float(sr)
        sig = (np.sin(phase) + 0.2 * np.sin(2.0 * phase)).astype(np.float32)
        sig = mix_background_noise(sig, _pink_noise(n, rng), snr_db=float(rng.uniform(5.0, 20.0)))
    elif choice < 0.90:
        # Acoustic impulse / gunshot / clap + ambient noise
        sig = 0.08 * _pink_noise(n, rng)
        n_impulses = int(rng.integers(1, 4))
        for _ in range(n_impulses):
            idx = int(rng.integers(int(0.05 * n), int(0.85 * n)))
            decay = np.exp(-np.arange(n - idx, dtype=np.float32) / (0.025 * sr))
            burst = rng.normal(0.0, 1.0, n - idx).astype(np.float32) * decay
            sig[idx:] += float(rng.uniform(0.6, 1.2)) * burst
        return np.clip(sig * float(rng.uniform(0.25, 0.8)), -1.0, 1.0).astype(np.float32)
    else:
        # Quiet ambient / near-silent sensor floor
        sig = rng.normal(0.0, float(rng.uniform(1e-4, 5e-3)), n).astype(np.float32)
        return np.clip(sig, -1.0, 1.0).astype(np.float32)

    return np.clip(sig * float(rng.uniform(0.35, 1.0)), -1.0, 1.0).astype(np.float32)


_CLASS_GENERATORS = {
    "Shahed_Piston_UAV": _generate_shahed_piston,
    "Multirotor_UAV": _generate_multirotor,
    "FixedWing_Electric_UAV": _generate_fixed_wing_electric,
    "Ground_Combustion_Noise": _generate_ground_combustion,
    "Environmental_Background": _generate_environmental_background,
}


def generate_clip_for_class(
    class_name: str,
    rng: np.random.Generator | None = None,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Generate a single 1-second float32 clip for one of the 5 target classes."""
    if class_name not in _CLASS_GENERATORS:
        raise ValueError(f"Unknown class '{class_name}'. Valid classes: {CLASSES}")
    gen = rng if rng is not None else np.random.default_rng()
    clip = _CLASS_GENERATORS[class_name](gen, sr=sr)
    if clip.size != NUM_SAMPLES:
        clip = np.resize(clip, NUM_SAMPLES).astype(np.float32)
    return clip


def generate_synthetic_dataset(
    num_samples_per_class: int = 300,
    seed: int = 42,
    sr: int = SAMPLE_RATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate balanced synthetic waveforms [N, 16000] and integer labels [N]."""
    rng = np.random.default_rng(seed)
    waveforms: list[np.ndarray] = []
    labels: list[int] = []

    for class_name in CLASSES:
        class_idx = CLASS_TO_IDX[class_name]
        for _ in range(num_samples_per_class):
            waveforms.append(generate_clip_for_class(class_name, rng=rng, sr=sr))
            labels.append(class_idx)

    x = np.stack(waveforms, axis=0).astype(np.float32, copy=False)
    y = np.asarray(labels, dtype=np.int64)
    perm = rng.permutation(x.shape[0])
    return x[perm], y[perm]


def export_synthetic_wavs(
    output_dir: Path | str,
    num_samples_per_class: int = 50,
    seed: int = 42,
) -> int:
    """Optionally materialize synthetic clips as 16kHz PCM_16 WAVs organized by class."""
    out_root = Path(output_dir)
    rng = np.random.default_rng(seed)
    total = 0
    for class_name in CLASSES:
        cls_dir = out_root / class_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(num_samples_per_class):
            clip = generate_clip_for_class(class_name, rng=rng, sr=SAMPLE_RATE)
            sf.write(cls_dir / f"{class_name.lower()}_{idx:04d}.wav", clip, SAMPLE_RATE, subtype="PCM_16")
            total += 1
    return total


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic 5-class UAV acoustic corpus.")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "synthetic")
    parser.add_argument("--num-samples-per-class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    count = export_synthetic_wavs(
        output_dir=args.output_dir,
        num_samples_per_class=args.num_samples_per_class,
        seed=args.seed,
    )
    print(f"Exported {count} synthetic 16kHz WAVs across {len(CLASSES)} classes to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
