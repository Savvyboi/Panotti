import { describe, expect, it } from 'vitest';
import { RingBufferResampler } from '../src/audio/AudioProcessorWorklet';
import { estimateSignalMetrics } from '../src/audio/MelSpectrogramExtractor';

describe('RingBufferResampler', () => {
  it('emits 16000-sample frames every 4000 samples once buffer fills at 16 kHz', () => {
    const resampler = new RingBufferResampler(16000, 16000, 16000, 4000);

    // Push in 128-sample WebAudio chunks up to 16000 samples (125 chunks = 16000)
    const emittedFrames: Float32Array[] = [];
    for (let offset = 0; offset < 24000; offset += 128) {
      const len = Math.min(128, 24000 - offset);
      const chunk = new Float32Array(len);
      for (let i = 0; i < len; i += 1) {
        chunk[i] = offset + i;
      }
      emittedFrames.push(...resampler.push(chunk));
    }

    // Should emit at 16000, 20000, 24000 -> 3 frames total
    expect(emittedFrames).toHaveLength(3);
    expect(emittedFrames[0].length).toBe(16000);
    expect(emittedFrames[0][0]).toBe(0);
    expect(emittedFrames[0][15999]).toBe(15999);

    expect(emittedFrames[1][0]).toBe(4000);
    expect(emittedFrames[1][15999]).toBe(19999);

    expect(emittedFrames[2][0]).toBe(8000);
    expect(emittedFrames[2][15999]).toBe(23999);
  });

  it('downsamples 48 kHz audio to 16 kHz preserving target tone frequency and suppressing aliasing', () => {
    const resampler = new RingBufferResampler(48000, 16000, 16000, 4000);

    // 1.5 seconds at 48 kHz = 72000 samples -> 24000 output samples at 16 kHz -> 3 frames
    const input48k = new Float32Array(72000);
    for (let i = 0; i < input48k.length; i += 1) {
      const t = i / 48000;
      input48k[i] = 0.5 * Math.sin(2 * Math.PI * 500 * t);
    }

    const frames: Float32Array[] = [];
    for (let i = 0; i < input48k.length; i += 128) {
      frames.push(...resampler.push(input48k.subarray(i, Math.min(i + 128, input48k.length))));
    }

    expect(frames.length).toBeGreaterThanOrEqual(2);
    const metrics = estimateSignalMetrics(frames[frames.length - 1]);
    expect(Math.abs(metrics.peak_freq_hz - 500)).toBeLessThan(12);
    expect(metrics.snr_db).toBeGreaterThan(15);
  });

  it('resamples 44.1 kHz audio to 16 kHz across arbitrary chunk sizes', () => {
    const resampler = new RingBufferResampler(44100, 16000, 16000, 4000);
    const totalIn = 66150; // 1.5s at 44.1 kHz (~24000 samples at 16 kHz)
    const input = new Float32Array(totalIn);
    for (let i = 0; i < totalIn; i += 1) {
      input[i] = 0.5 * Math.sin((2 * Math.PI * 440 * i) / 44100);
    }

    const frames = resampler.push(input);
    expect(frames.length).toBeGreaterThanOrEqual(2);
    const metrics = estimateSignalMetrics(frames[frames.length - 1]);
    expect(Math.abs(metrics.peak_freq_hz - 440)).toBeLessThan(12);
  });

  it('resets state cleanly', () => {
    const resampler = new RingBufferResampler(16000, 16000, 16000, 4000);
    resampler.push(new Float32Array(16000).fill(1));
    expect(resampler.getTotalOutputSamples()).toBe(16000);
    resampler.reset();
    expect(resampler.getTotalOutputSamples()).toBe(0);
    expect(resampler.push(new Float32Array(4000).fill(2))).toHaveLength(0);
  });
});
