import { describe, expect, it } from 'vitest';
import {
  buildMelFilterbank,
  buildPeriodicHannWindow,
  DSP_CONFIG,
  estimateSignalMetrics,
  hzToMel,
  MelSpectrogramExtractor,
  melToHz,
} from '../src/audio/MelSpectrogramExtractor';
import { synthesizeSimulationFrame } from '../src/audio/StreamManager';

function naiveReferenceLogMel(pcm16k: Float32Array): Float32Array {
  const { sampleRate, nFft, hopLength, numFrames, nMels, fMin, fMax, paddedLength, numBins, eps } =
    DSP_CONFIG;
  const padded = new Float64Array(paddedLength);
  for (let i = 0; i < Math.min(pcm16k.length, sampleRate); i += 1) {
    padded[i] = pcm16k[i];
  }

  const win = buildPeriodicHannWindow(nFft);
  const fb = buildMelFilterbank(nMels, nFft, sampleRate, fMin, fMax);
  const logMel = new Float64Array(nMels * numFrames);

  let sum = 0;
  for (let t = 0; t < numFrames; t += 1) {
    const start = t * hopLength;
    const power = new Float64Array(numBins);

    for (let k = 0; k < numBins; k += 1) {
      let re = 0;
      let im = 0;
      for (let n = 0; n < nFft; n += 1) {
        const x = padded[start + n] * win[n];
        const angle = (-2 * Math.PI * k * n) / nFft;
        re += x * Math.cos(angle);
        im += x * Math.sin(angle);
      }
      power[k] = re * re + im * im;
    }

    for (let m = 0; m < nMels; m += 1) {
      let energy = 0;
      for (let k = 0; k < numBins; k += 1) {
        energy += fb[m * numBins + k] * power[k];
      }
      const val = Math.log(Math.max(energy, eps));
      logMel[m * numFrames + t] = val;
      sum += val;
    }
  }

  const total = nMels * numFrames;
  const mean = sum / total;
  let sq = 0;
  for (let i = 0; i < total; i += 1) {
    const d = logMel[i] - mean;
    sq += d * d;
  }
  const std = Math.sqrt(sq / total);
  const out = new Float32Array(total);
  for (let i = 0; i < total; i += 1) {
    out[i] = (logMel[i] - mean) / (std + eps);
  }
  return out;
}

describe('MelSpectrogramExtractor', () => {
  it('constructs a 128x257 HTK Mel filterbank with narrow-band fix and L1-normalized rows', () => {
    const fb = buildMelFilterbank();
    expect(fb.length).toBe(128 * 257);

    // Verify HTK scale equations match contract
    expect(hzToMel(0)).toBeCloseTo(0, 10);
    expect(melToHz(hzToMel(1000))).toBeCloseTo(1000, 8);

    for (let m = 0; m < 128; m += 1) {
      let rowSum = 0;
      for (let k = 0; k < 257; k += 1) {
        const w = fb[m * 257 + k];
        expect(w).toBeGreaterThanOrEqual(0);
        rowSum += w;
      }
      // Narrow-band interpolation ensures NO empty zero-weight rows at low frequencies
      expect(rowSum).toBeCloseTo(1.0, 9);
    }
  });

  it('matches an independent direct DFT reference spectrogram within 1e-5', () => {
    const signal = new Float32Array(16000);
    for (let i = 0; i < signal.length; i += 1) {
      const t = i / 16000;
      signal[i] =
        0.5 * Math.sin(2 * Math.PI * 92.5 * t) +
        0.3 * Math.sin(2 * Math.PI * 440 * t) +
        0.2 * Math.sin(2 * Math.PI * 1650 * t);
    }

    const extractor = new MelSpectrogramExtractor();
    const fast = extractor.extract(signal);
    const reference = naiveReferenceLogMel(signal);

    expect(fast).toBeInstanceOf(Float32Array);
    expect(fast.length).toBe(128 * 100);

    let maxDiff = 0;
    for (let i = 0; i < fast.length; i += 1) {
      const diff = Math.abs(fast[i] - reference[i]);
      if (diff > maxDiff) maxDiff = diff;
    }
    expect(maxDiff).toBeLessThan(1e-5);

    // Z-score mean ~= 0 and std ~= 1
    let mean = 0;
    for (let i = 0; i < fast.length; i += 1) mean += fast[i];
    mean /= fast.length;
    let varSum = 0;
    for (let i = 0; i < fast.length; i += 1) {
      const d = fast[i] - mean;
      varSum += d * d;
    }
    const std = Math.sqrt(varSum / fast.length);

    expect(Math.abs(mean)).toBeLessThan(1e-5);
    expect(std).toBeCloseTo(1.0, 3);
  });

  it('handles silent inputs without NaN or Infinity', () => {
    const extractor = new MelSpectrogramExtractor();
    const silence = new Float32Array(16000);
    const out = extractor.extract(silence);
    expect(out.length).toBe(12800);
    for (let i = 0; i < out.length; i += 1) {
      expect(Number.isFinite(out[i])).toBe(true);
    }

    const metrics = estimateSignalMetrics(silence);
    expect(metrics.snr_db).toBe(-30);
    expect(metrics.peak_freq_hz).toBe(0);
  });

  it('estimates peak frequency and SNR accurately for tones and Shahed simulation', () => {
    const tone1k = new Float32Array(16000);
    for (let i = 0; i < tone1k.length; i += 1) {
      tone1k[i] = 0.4 * Math.sin((2 * Math.PI * 1000 * i) / 16000);
    }
    const toneMetrics = estimateSignalMetrics(tone1k);
    expect(Math.abs(toneMetrics.peak_freq_hz - 1000)).toBeLessThan(8);
    expect(toneMetrics.snr_db).toBeGreaterThan(20);

    const shahedFrame = synthesizeSimulationFrame('shahed', 0);
    const shahedMetrics = estimateSignalMetrics(shahedFrame);
    expect(shahedMetrics.peak_freq_hz).toBeGreaterThan(80);
    expect(shahedMetrics.peak_freq_hz).toBeLessThan(400);
    expect(shahedMetrics.snr_db).toBeGreaterThan(10);

    const multirotorFrame = synthesizeSimulationFrame('multirotor', 0.25);
    const multirotorMetrics = estimateSignalMetrics(multirotorFrame);
    expect(multirotorFrame.length).toBe(16000);
    expect(multirotorMetrics.peak_freq_hz).toBeGreaterThan(140);
    expect(multirotorMetrics.snr_db).toBeGreaterThan(8);

    const bgFrame = synthesizeSimulationFrame('background', 0.5);
    const bgMetrics = estimateSignalMetrics(bgFrame);
    expect(bgFrame.length).toBe(16000);
    expect(bgMetrics.snr_db).toBeLessThan(shahedMetrics.snr_db);
  });
});
