"""Exact shared Log-Mel Spectrogram DSP pipeline matching apps/web/src/audio/MelSpectrogramExtractor.ts."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from config import (
    EPS,
    FMAX,
    FMIN,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    NUM_FRAMES,
    NUM_SAMPLES,
    PADDED_LENGTH,
    SAMPLE_RATE,
)


def hz_to_mel(freq_hz: np.ndarray | float) -> np.ndarray | float:
    """Convert frequency in Hz to HTK Mel scale."""
    return 2595.0 * np.log10(1.0 + np.asarray(freq_hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    """Convert HTK Mel scale to frequency in Hz."""
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


@lru_cache(maxsize=4)
def get_periodic_hann_window(n_fft: int = N_FFT) -> np.ndarray:
    """Compute exact periodic Hann window w[n] = 0.5 * (1 - cos(2 * pi * n / N_FFT))."""
    n = np.arange(n_fft, dtype=np.float64)
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * n / float(n_fft)))


@lru_cache(maxsize=4)
def build_mel_filterbank(
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
    fmin: float = FMIN,
    fmax: float = FMAX,
) -> np.ndarray:
    """Construct [n_mels, n_fft // 2 + 1] HTK triangular Mel filterbank with narrow-band fix."""
    n_bins = n_fft // 2 + 1
    df = float(sample_rate) / float(n_fft)
    bin_freqs = np.arange(n_bins, dtype=np.float64) * df

    mel_min = float(hz_to_mel(fmin))
    mel_max = float(hz_to_mel(fmax))
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2, dtype=np.float64)
    hz_points = np.asarray(mel_to_hz(mel_points), dtype=np.float64)

    filterbank = np.zeros((n_mels, n_bins), dtype=np.float64)

    for m in range(n_mels):
        f_left = hz_points[m]
        f_center = hz_points[m + 1]
        f_right = hz_points[m + 2]

        weights = np.zeros(n_bins, dtype=np.float64)
        rising = (bin_freqs >= f_left) & (bin_freqs <= f_center)
        if f_center > f_left:
            weights[rising] = (bin_freqs[rising] - f_left) / (f_center - f_left)

        falling = (bin_freqs > f_center) & (bin_freqs <= f_right)
        if f_right > f_center:
            weights[falling] = (f_right - bin_freqs[falling]) / (f_right - f_center)

        row_sum = float(np.sum(weights))
        if row_sum == 0.0:
            # Narrow-band fix: linearly interpolate between adjacent FFT bins around f_center
            pos = f_center / df
            k0 = int(np.floor(pos))
            frac = pos - float(k0)
            k0 = int(np.clip(k0, 0, n_bins - 1))
            k1 = min(k0 + 1, n_bins - 1)
            if k0 == k1:
                weights[k0] = 1.0
            else:
                weights[k0] = 1.0 - frac
                weights[k1] = frac
            row_sum = float(np.sum(weights))

        if row_sum > 0.0:
            weights /= row_sum

        filterbank[m] = weights

    return filterbank


def normalize_clip_length(audio: np.ndarray, num_samples: int = NUM_SAMPLES) -> np.ndarray:
    """Convert input to mono float32 and pad/truncate to exact num_samples."""
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim > 1:
        arr = np.mean(arr, axis=-1, dtype=np.float32)
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if arr.size < num_samples:
        return np.pad(arr, (0, num_samples - arr.size)).astype(np.float32, copy=False)
    if arr.size > num_samples:
        return arr[:num_samples].astype(np.float32, copy=False)
    return arr


def extract_log_mel_spectrogram(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Compute exact shared [1, 128, 100] float32 normalized Log-Mel spectrogram.

    Contract:
    - 16000 samples mono float32 (padded/truncated), right-padded to 16352 samples
    - 100 frames of 512 samples with hop 160
    - Periodic Hann window, rfft power spectrum |FFT|^2 (257 bins)
    - 128-bin HTK Mel filterbank (50..4000 Hz, narrow-band interp fix, L1 row norm)
    - ln(max(mel_energy, 1e-6)) followed by per-spectrogram z-score (S - mean) / (std + 1e-6)
    """
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected sample rate {SAMPLE_RATE} Hz, got {sr} Hz.")

    clip = normalize_clip_length(audio, NUM_SAMPLES)
    padded = np.zeros(PADDED_LENGTH, dtype=np.float32)
    padded[:NUM_SAMPLES] = clip

    # Strided framing for t = 0..99: padded[t*160 : t*160 + 512]
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP_LENGTH][:NUM_FRAMES]
    window = get_periodic_hann_window(N_FFT)
    windowed = frames.astype(np.float64) * window.astype(np.float64)

    fft_vals = np.fft.rfft(windowed, n=N_FFT, axis=-1)
    power_spec = (fft_vals.real ** 2 + fft_vals.imag ** 2).astype(np.float64)  # [100, 257]

    mel_fb = build_mel_filterbank(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
    ).astype(np.float64)

    mel_energy = mel_fb @ power_spec.T  # [128, 100]
    log_mel = np.log(np.maximum(mel_energy, EPS))

    mean = float(np.mean(log_mel))
    std = float(np.std(log_mel))
    norm_spec = (log_mel - mean) / (std + EPS)

    return norm_spec[np.newaxis, :, :].astype(np.float32, copy=False)


def extract_batch_spectrograms(
    clips: np.ndarray | list[np.ndarray],
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Extract batched [B, 1, 128, 100] float32 Log-Mel spectrograms."""
    specs = [extract_log_mel_spectrogram(clip, sr=sr) for clip in clips]
    if not specs:
        return np.zeros((0, 1, N_MELS, NUM_FRAMES), dtype=np.float32)
    return np.stack(specs, axis=0).astype(np.float32, copy=False)


def compute_peak_freq_and_snr(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> tuple[float, float]:
    """Estimate dominant spectral peak frequency (Hz) and broadband SNR (dB)."""
    clip = normalize_clip_length(audio, min(max(int(sr), 1), max(len(audio), int(sr))))
    if clip.size < N_FFT or float(np.max(np.abs(clip))) < 1e-8:
        return 0.0, -30.0

    # Average power spectrum over overlapping 1024-sample Hann frames for clean resolution (15.625 Hz/bin)
    n_fft = 1024 if clip.size >= 1024 else N_FFT
    hop = n_fft // 4
    n_frames = 1 + (clip.size - n_fft) // hop
    frames = np.lib.stride_tricks.sliding_window_view(clip, n_fft)[::hop][:n_frames]
    window = get_periodic_hann_window(n_fft).astype(np.float64)
    fft_vals = np.fft.rfft(frames.astype(np.float64) * window, n=n_fft, axis=-1)
    psd = np.mean(fft_vals.real ** 2 + fft_vals.imag ** 2, axis=0)

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sr))
    valid_mask = (freqs >= FMIN) & (freqs <= float(sr) * 0.5)
    valid_psd = psd[valid_mask]
    valid_freqs = freqs[valid_mask]
    if valid_psd.size == 0:
        return 0.0, -30.0

    peak_local = int(np.argmax(valid_psd))
    peak_freq = float(valid_freqs[peak_local])

    # Parabolic interpolation around peak bin for sub-bin frequency precision
    if 0 < peak_local < valid_psd.size - 1:
        y0 = float(np.log(max(valid_psd[peak_local - 1], 1e-12)))
        y1 = float(np.log(max(valid_psd[peak_local], 1e-12)))
        y2 = float(np.log(max(valid_psd[peak_local + 1], 1e-12)))
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-12:
            delta = float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))
            df = float(sr) / float(n_fft)
            peak_freq += delta * df

    noise_floor = float(np.median(valid_psd))
    noise_total = max(noise_floor * float(valid_psd.size), 1e-12)
    signal_excess = max(float(np.sum(np.maximum(valid_psd - noise_floor, 0.0))), 1e-12)
    snr_db = float(np.clip(10.0 * np.log10(signal_excess / noise_total), -30.0, 60.0))

    return float(peak_freq), float(snr_db)
