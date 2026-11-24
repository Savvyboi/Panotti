"""Comprehensive tests for Panotti dataset builder, DSP contract, physics augmentations, training, and ONNX export."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import soundfile as sf
import torch

# Ensure pipelines/dataset_builder is importable regardless of pytest invocation cwd
BUILDER_DIR = Path(__file__).resolve().parent.parent
if str(BUILDER_DIR) not in sys.path:
    sys.path.insert(0, str(BUILDER_DIR))

import clean_and_filter
import config
import export_onnx
import feature_extraction
import model as model_mod
import physics_augment
import scrape_osint
import synthetic_dataset
import train as train_mod


def _make_harmonic_comb(f0_hz: float, sr: int = 16000, duration_s: float = 1.0, max_hz: float = 3500.0) -> np.ndarray:
    t = np.arange(int(sr * duration_s), dtype=np.float64) / float(sr)
    sig = np.zeros_like(t)
    k = 1
    while k * f0_hz <= max_hz:
        sig += (1.0 / (k ** 0.8)) * np.sin(2.0 * np.pi * k * f0_hz * t)
        k += 1
    sig /= np.max(np.abs(sig)) + 1e-12
    return (0.6 * sig).astype(np.float32)


def test_config_constants_and_classes() -> None:
    assert config.SAMPLE_RATE == 16000
    assert config.N_FFT == 512
    assert config.HOP_LENGTH == 160
    assert config.NUM_FRAMES == 100
    assert config.PADDED_LENGTH == 16352
    assert config.N_MELS == 128
    assert config.FMIN == 50.0
    assert config.FMAX == 4000.0
    assert config.CLASSES == [
        "Shahed_Piston_UAV",
        "Multirotor_UAV",
        "FixedWing_Electric_UAV",
        "Ground_Combustion_Noise",
        "Environmental_Background",
    ]


def test_scrape_osint_dry_run_manifest(tmp_path: Path) -> None:
    out_dir = tmp_path / "raw"
    rc = scrape_osint.main(["--dry-run", "--output-dir", str(out_dir), "--max-per-query", "2"])
    assert rc == 0

    manifest_path = out_dir / "manifest.json"
    archive_path = out_dir / "download_archive.txt"
    assert manifest_path.exists()
    assert archive_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["dry_run"] is True
    assert manifest["audio_format"]["sample_rate"] == 16000
    assert manifest["audio_format"]["channels"] == 1
    assert manifest["audio_format"]["subtype"] == "PCM_16"
    assert manifest["queries"] == config.DEFAULT_OSINT_QUERIES
    assert manifest["max_duration_sec"] == 300
    assert len(manifest["entries"]) == len(config.DEFAULT_OSINT_QUERIES)
    assert all(entry["status"] == "dry_run_planned" for entry in manifest["entries"])


def test_scrape_osint_duration_and_live_filter(tmp_path: Path) -> None:
    opts = scrape_osint.build_yt_dlp_options(
        query_dir=tmp_path,
        archive_path=tmp_path / "archive.txt",
        max_duration_sec=300,
    )
    mf = opts["match_filter"]
    assert mf({"duration": 45, "is_live": False, "title": "Shahed 136 flyby"}) is None
    assert mf({"duration": 300, "is_live": False, "title": "DLE 111 boxer run"}) is None
    assert mf({"duration": 301, "is_live": False, "title": "Long video"}) is not None
    assert mf({"duration": 20, "is_live": True, "title": "Live stream"}) is not None
    assert mf({"duration": 112, "is_live": False, "title": "Russian Song About Geran 2"}) is not None
    assert mf({"duration": 36, "is_live": False, "title": "shahed 136 ocarina sound"}) is not None
    assert mf({"duration": 60, "is_live": False, "title": "White Noise 10 Hours"}) is not None


def test_class_presets_and_cleaning_profiles_cover_all_classes() -> None:
    assert set(config.CLASS_OSINT_QUERIES.keys()) == set(config.CLASSES)
    assert set(config.CLASS_CLEANING_PROFILES.keys()) == set(config.CLASSES)
    for cls_name, queries in config.CLASS_OSINT_QUERIES.items():
        assert len(queries) >= 5, f"{cls_name} should have >= 5 OSINT queries"


def test_isolate_non_vocal_spectral_fallback() -> None:
    rng = np.random.default_rng(7)
    sr = 16000
    t = np.arange(sr, dtype=np.float32) / float(sr)

    # Broad speech-formant noise inside 400-2800 Hz
    raw = rng.normal(0.0, 1.0, sr).astype(np.float32)
    spec = np.fft.rfft(raw)
    freqs = np.fft.rfftfreq(sr, d=1.0 / sr)
    speech_band = (freqs >= 400.0) & (freqs <= 2800.0)
    speech_noise = np.fft.irfft(spec * speech_band.astype(np.float32), n=sr).astype(np.float32)

    cleaned_speech = clean_and_filter.isolate_non_vocal(speech_noise, sr=sr, backend="spectral")
    assert cleaned_speech.shape == speech_noise.shape
    assert cleaned_speech.dtype == np.float32
    assert float(np.linalg.norm(cleaned_speech)) < 0.55 * float(np.linalg.norm(speech_noise))

    # UAV harmonic comb should retain substantial energy (especially low fundamentals)
    comb = _make_harmonic_comb(92.5, sr=sr)
    cleaned_comb = clean_and_filter.isolate_non_vocal(comb, sr=sr, backend="auto")
    assert cleaned_comb.shape == comb.shape
    assert float(np.linalg.norm(cleaned_comb)) > 0.65 * float(np.linalg.norm(comb))


def test_siren_gunfire_filter_pluggable_and_heuristic() -> None:
    sr = 16000
    frame = np.zeros(sr, dtype=np.float32)

    # Pluggable scorer boundary checks (> 0.30 rejects, <= 0.30 accepts)
    for label in config.SIREN_GUNFIRE_LABELS:
        filt_reject = clean_and_filter.SirenGunfireFilter(
            threshold=0.30,
            scorer=lambda _f, _sr, lbl=label: {lbl: 0.31},
        )
        assert filt_reject.should_reject(frame, sr) is True

        filt_boundary = clean_and_filter.SirenGunfireFilter(
            threshold=0.30,
            scorer=lambda _f, _sr, lbl=label: {lbl: 0.30},
        )
        assert filt_boundary.should_reject(frame, sr) is False

    # Heuristic fallback checks
    heuristic_filter = clean_and_filter.SirenGunfireFilter(threshold=0.30, backend="heuristic")
    t = np.arange(sr, dtype=np.float64) / float(sr)

    # Sweeping FM siren (700 -> 1500 Hz)
    f_inst = 1100.0 + 400.0 * np.sin(2.0 * np.pi * 1.5 * t)
    siren = np.sin(2.0 * np.pi * np.cumsum(f_inst) / float(sr)).astype(np.float32)
    assert heuristic_filter.should_reject(siren, sr) is True

    # Sharp impulse / gunshot
    rng = np.random.default_rng(11)
    gunshot = np.zeros(sr, dtype=np.float32)
    gunshot[4000:] = rng.normal(0.0, 1.0, sr - 4000).astype(np.float32) * np.exp(
        -np.arange(sr - 4000, dtype=np.float32) / (0.015 * sr)
    )
    assert heuristic_filter.should_reject(gunshot, sr) is True

    # Clean Shahed 92.5 Hz comb must NOT be rejected
    shahed = _make_harmonic_comb(92.5, sr=sr)
    assert heuristic_filter.should_reject(shahed, sr) is False


def test_verify_harmonic_comb() -> None:
    sr = 16000
    rng = np.random.default_rng(19)

    assert clean_and_filter.verify_harmonic_comb(_make_harmonic_comb(92.5, sr=sr), sr=sr) is True
    assert clean_and_filter.verify_harmonic_comb(_make_harmonic_comb(180.0, sr=sr), sr=sr) is True

    # Out-of-band frequencies and non-periodic noise should be rejected
    assert clean_and_filter.verify_harmonic_comb(_make_harmonic_comb(30.0, sr=sr), sr=sr) is False
    assert clean_and_filter.verify_harmonic_comb(_make_harmonic_comb(500.0, sr=sr), sr=sr) is False
    assert clean_and_filter.verify_harmonic_comb(rng.normal(0.0, 0.2, sr).astype(np.float32), sr=sr) is False
    assert clean_and_filter.verify_harmonic_comb(np.zeros(sr, dtype=np.float32), sr=sr) is False


def test_verify_harmonic_comb_ignores_main_lobe_on_lowpassed_piston() -> None:
    sr = 16000
    # Low-pass filtered 92.5 Hz comb mimics muffled piston/rotary exhaust with high r[8] main lobe
    raw_comb = _make_harmonic_comb(92.5, sr=sr, max_hz=600.0)
    lowpassed = physics_augment.apply_atmospheric_absorption(raw_comb, sr=sr, distance_m=2500.0)
    r = clean_and_filter._normalized_autocorrelation(lowpassed, 400)
    assert float(r[8]) > 0.7
    assert clean_and_filter.verify_harmonic_comb(lowpassed, sr=sr, min_hz=50.0, max_hz=300.0) is True
    assert clean_and_filter.verify_harmonic_comb(_make_harmonic_comb(500.0, sr=sr), sr=sr, min_hz=50.0, max_hz=300.0) is False


def test_clean_directory_cli(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    clean_dir = tmp_path / "clean"
    raw_dir.mkdir()

    sr = 16000
    t = np.arange(sr, dtype=np.float64) / float(sr)
    shahed = _make_harmonic_comb(92.5, sr=sr)
    f_inst = 1100.0 + 400.0 * np.sin(2.0 * np.pi * 1.5 * t)
    siren = (0.6 * np.sin(2.0 * np.pi * np.cumsum(f_inst) / float(sr))).astype(np.float32)
    noise = np.random.default_rng(3).normal(0.0, 0.1, sr).astype(np.float32)

    sf.write(raw_dir / "shahed.wav", shahed, sr, subtype="PCM_16")
    sf.write(raw_dir / "siren.wav", siren, sr, subtype="PCM_16")
    sf.write(raw_dir / "noise.wav", noise, sr, subtype="PCM_16")

    rc = clean_and_filter.main(
        ["--input-dir", str(raw_dir), "--output-dir", str(clean_dir), "--vocal-backend", "spectral"]
    )
    assert rc == 0

    report = json.loads((clean_dir / "clean_report.json").read_text(encoding="utf-8"))
    assert report["source_files"] == 3
    assert report["kept_frames"] == 1
    assert report["rejected_siren_gunfire"] == 1
    assert report["rejected_harmonic_comb"] == 1

    kept_wav = Path(report["kept_files"][0])
    info = sf.info(kept_wav)
    assert info.samplerate == 16000
    assert info.channels == 1
    assert info.subtype == "PCM_16"


def test_clean_directory_class_profiles_multirotor_and_background(tmp_path: Path) -> None:
    sr = 16000
    t = np.arange(sr, dtype=np.float64) / float(sr)

    # 450 Hz multirotor blade-pass harmonic comb is rejected by [50, 300] but kept by Multirotor_UAV [80, 900]
    mr_dir = tmp_path / "raw_mr"
    mr_dir.mkdir()
    mr_sig = _make_harmonic_comb(450.0, sr=sr)
    sf.write(mr_dir / "fpv.wav", mr_sig, sr, subtype="PCM_16")

    out_mr = tmp_path / "clean_mr"
    rc_mr = clean_and_filter.main(
        [
            "--input-dir",
            str(mr_dir),
            "--output-dir",
            str(out_mr),
            "--class-profile",
            "Multirotor_UAV",
            "--vocal-backend",
            "spectral",
        ]
    )
    assert rc_mr == 0
    mr_report = json.loads((out_mr / "clean_report.json").read_text(encoding="utf-8"))
    assert mr_report["kept_frames"] == 1
    assert mr_report["comb_max_hz"] == 900.0

    # Environmental_Background disables both comb check and siren rejection so sirens stay in negatives
    bg_dir = tmp_path / "raw_bg"
    bg_dir.mkdir()
    f_inst = 1100.0 + 400.0 * np.sin(2.0 * np.pi * 1.5 * t)
    siren = (0.6 * np.sin(2.0 * np.pi * np.cumsum(f_inst) / float(sr))).astype(np.float32)
    sf.write(bg_dir / "siren_bg.wav", siren, sr, subtype="PCM_16")

    out_bg = tmp_path / "clean_bg"
    rc_bg = clean_and_filter.main(
        [
            "--input-dir",
            str(bg_dir),
            "--output-dir",
            str(out_bg),
            "--class-profile",
            "Environmental_Background",
            "--vocal-backend",
            "spectral",
        ]
    )
    assert rc_bg == 0
    bg_report = json.loads((out_bg / "clean_report.json").read_text(encoding="utf-8"))
    assert bg_report["kept_frames"] == 1
    assert bg_report["apply_siren_filter"] is False


def test_physics_augmentations() -> None:
    sr = 16000
    t = np.arange(sr, dtype=np.float64) / float(sr)

    # 1. Atmospheric absorption: attenuates high frequencies much more than low frequencies
    low_tone = np.sin(2.0 * np.pi * 100.0 * t).astype(np.float32)
    high_tone = np.sin(2.0 * np.pi * 3500.0 * t).astype(np.float32)
    low_far = physics_augment.apply_atmospheric_absorption(low_tone, sr=sr, distance_m=2500.0)
    high_far = physics_augment.apply_atmospheric_absorption(high_tone, sr=sr, distance_m=2500.0)
    high_near = physics_augment.apply_atmospheric_absorption(high_tone, sr=sr, distance_m=100.0)

    assert float(np.linalg.norm(low_far)) > 0.90 * float(np.linalg.norm(low_tone))
    assert float(np.linalg.norm(high_far)) < 0.25 * float(np.linalg.norm(high_near))

    # 2. Doppler shift: approaching (t < 0.5) higher pitch, receding (t > 0.5) lower pitch
    carrier = np.sin(2.0 * np.pi * 500.0 * t).astype(np.float32)
    warped = physics_augment.apply_doppler_shift(
        carrier, sr=sr, velocity_mps=50.0, min_distance_m=150.0, t_center=0.5
    )
    f_approach, _ = feature_extraction.compute_peak_freq_and_snr(warped[:6000], sr=sr)
    f_recede, _ = feature_extraction.compute_peak_freq_and_snr(warped[10000:], sr=sr)
    assert f_approach > 505.0
    assert f_recede < 495.0

    # 3. Blade flutter: amplitude envelope oscillates at f_b with depth m
    dc = np.ones(sr, dtype=np.float32)
    fluttered = physics_augment.apply_blade_flutter(dc, sr=sr, f_b=25.0, depth=0.12)
    assert np.isclose(float(np.max(fluttered)), 1.12, atol=1e-3)
    assert np.isclose(float(np.min(fluttered)), 0.88, atol=1e-3)

    # 4. SNR mixing: exact energy ratio verification
    rng = np.random.default_rng(9)
    sig = low_tone
    noise = rng.normal(0.0, 1.0, sr).astype(np.float32)
    mixed = physics_augment.mix_background_noise(sig, noise, snr_db=10.0)
    measured_noise = mixed - sig
    actual_snr = 20.0 * np.log10(np.linalg.norm(sig) / np.linalg.norm(measured_noise))
    assert np.isclose(actual_snr, 10.0, atol=1e-3)


def test_class_aware_physics_augment() -> None:
    sr = 16000
    tone = np.sin(2.0 * np.pi * 440.0 * np.arange(sr, dtype=np.float64) / float(sr)).astype(np.float32)
    rng = np.random.default_rng(12)

    bg_aug, bg_params = physics_augment.augment_clip(
        tone, sr=sr, rng=rng, class_name="Environmental_Background"
    )
    assert bg_aug.shape == tone.shape
    assert bg_params["velocity_mps"] == 0.0
    assert bg_params["flutter_hz"] is None

    gc_aug, gc_params = physics_augment.augment_clip(
        tone, sr=sr, rng=rng, class_name="Ground_Combustion_Noise"
    )
    assert gc_aug.shape == tone.shape
    assert gc_params["velocity_mps"] == 0.0
    assert gc_params["flutter_hz"] is None

    # Multirotor should enable flutter on at least one draw out of 10
    mr_flutters = [
        physics_augment.augment_clip(tone, sr=sr, rng=rng, class_name="Multirotor_UAV")[1]["flutter_hz"]
        for _ in range(10)
    ]
    assert any(f is not None for f in mr_flutters)


def test_grouped_split_prevents_video_leakage(tmp_path: Path) -> None:
    clean_dir = tmp_path / "clean"
    aug_dir = tmp_path / "augmented"
    sr = 16000
    sig = _make_harmonic_comb(92.5, sr=sr)

    # Create 3 distinct video IDs per class (2 frames each)
    for cls_name in config.CLASSES:
        (clean_dir / cls_name).mkdir(parents=True, exist_ok=True)
        (aug_dir / cls_name).mkdir(parents=True, exist_ok=True)
        for vid in ("vidA", "vidB", "vidC"):
            for f_idx in range(2):
                sf.write(
                    clean_dir / cls_name / f"{vid}_frame_{f_idx:04d}.wav",
                    sig,
                    sr,
                    subtype="PCM_16",
                )
                sf.write(
                    aug_dir / cls_name / f"{vid}_frame_{f_idx:04d}_aug_00.wav",
                    sig,
                    sr,
                    subtype="PCM_16",
                )

    _, labels, groups = train_mod.load_wav_directory_with_groups(clean_dir)
    train_idx, val_idx, train_grps, val_grps = train_mod.grouped_stratified_split(
        labels, groups, val_fraction=0.33, seed=42
    )
    assert train_idx.size > 0 and val_idx.size > 0
    for cls_idx in range(len(config.CLASSES)):
        assert train_grps[cls_idx].isdisjoint(val_grps[cls_idx])
        assert len(val_grps[cls_idx]) == 1

    # Train 1 epoch with augmented_dir: num_train_samples must exclude val video IDs
    # Per class: 2 train vids * 2 clean frames + 2 train vids * 2 aug frames = 8 train samples -> 40 total
    metrics = train_mod.train_model(
        epochs=1,
        batch_size=16,
        use_synthetic=False,
        data_dir=clean_dir,
        augmented_dir=aug_dir,
        output_ckpt=tmp_path / "ckpt.pt",
        seed=42,
    )
    assert metrics["split_strategy"] == "grouped_by_source_video_id"
    assert metrics["num_val_samples"] == 10  # 5 classes * 1 val vid * 2 frames
    assert metrics["num_train_samples"] == 40  # 5 classes * 2 train vids * (2 clean + 2 aug)


def test_dsp_contract_shape_and_values() -> None:
    fb = feature_extraction.build_mel_filterbank()
    assert fb.shape == (128, 257)
    row_sums = fb.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)

    # Padding/truncation handling and exact [1, 128, 100] z-scored output
    comb = _make_harmonic_comb(92.5, sr=16000)
    spec_exact = feature_extraction.extract_log_mel_spectrogram(comb, sr=16000)
    spec_short = feature_extraction.extract_log_mel_spectrogram(comb[:8000], sr=16000)
    spec_long = feature_extraction.extract_log_mel_spectrogram(np.concatenate([comb, comb]), sr=16000)

    for s in (spec_exact, spec_short, spec_long):
        assert s.dtype == np.float32
        assert s.shape == (1, 128, 100)
        assert np.all(np.isfinite(s))
        assert abs(float(np.mean(s))) < 1e-4
        assert abs(float(np.std(s)) - 1.0) < 1e-3

    # Peak frequency & SNR estimation
    peak_hz, snr_high = feature_extraction.compute_peak_freq_and_snr(comb, sr=16000)
    assert 85.0 <= peak_hz <= 100.0
    assert snr_high > 5.0

    noisy_comb = physics_augment.mix_background_noise(
        comb, np.random.default_rng(1).normal(0.0, 1.0, 16000).astype(np.float32), snr_db=0.0
    )
    _, snr_low = feature_extraction.compute_peak_freq_and_snr(noisy_comb, sr=16000)
    assert snr_high > snr_low


def test_synthetic_dataset_and_model_forward() -> None:
    waves, labels = synthetic_dataset.generate_synthetic_dataset(num_samples_per_class=4, seed=42)
    assert waves.shape == (20, 16000)
    assert waves.dtype == np.float32
    assert set(labels.tolist()) == {0, 1, 2, 3, 4}

    specs = feature_extraction.extract_batch_spectrograms(waves[:4])
    assert specs.shape == (4, 1, 128, 100)

    net = model_mod.AcousticMobileNet(num_classes=5)
    net.eval()
    with torch.no_grad():
        logits = net(torch.from_numpy(specs))
    assert logits.shape == (4, 5)
    assert torch.all(torch.isfinite(logits))


def test_training_convergence_and_onnx_export_parity(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "checkpoints" / "best_model.pt"
    metrics = train_mod.train_model(
        epochs=2,
        batch_size=16,
        use_synthetic=True,
        num_samples_per_class=20,
        output_ckpt=ckpt_path,
        seed=42,
    )
    assert ckpt_path.exists()
    assert (ckpt_path.parent / "metrics.json").exists()
    assert len(metrics["history"]) == 2
    assert np.isfinite(metrics["history"][-1]["train_loss"])

    fp32_path = tmp_path / "uav_fp32.onnx"
    int8_path = tmp_path / "uav_int8.onnx"
    meta_path = tmp_path / "model_metadata.json"

    metadata = export_onnx.export_quantized_onnx(
        checkpoint_path=ckpt_path,
        fp32_output=fp32_path,
        int8_output=int8_path,
        metadata_output=meta_path,
    )

    assert fp32_path.exists()
    assert int8_path.exists()
    assert meta_path.exists()
    assert int8_path.stat().st_size < config.MAX_ONNX_BYTES
    assert metadata["classes"] == config.CLASSES
    assert metadata["parity"]["fp32_max_abs_diff"] < 1e-4

    # Verify dynamic batch dimension (batch=3) on exported INT8 ONNX model
    sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    batch_in = np.random.default_rng(5).normal(0.0, 1.0, (3, 1, 128, 100)).astype(np.float32)
    out = sess.run([config.ONNX_OUTPUT_NAME], {config.ONNX_INPUT_NAME: batch_in})[0]
    assert out.shape == (3, 5)
    assert np.all(np.isfinite(out))
