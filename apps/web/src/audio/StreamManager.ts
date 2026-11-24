import type { AudioCaptureState, SimulationProfile } from '../types';
import { HOP_SIZE_SAMPLES, TARGET_SAMPLE_RATE, WINDOW_SIZE_SAMPLES } from './AudioProcessorWorklet';
import workletUrl from './AudioProcessorWorklet.ts?worker&url';

export interface StreamManagerCallbacks {
  onFrame: (frame16k: Float32Array) => void;
  onStateChange?: (state: AudioCaptureState, sampleRate: number) => void;
  onError?: (message: string) => void;
}

/**
 * Generates deterministic-quality realistic 16 kHz acoustic frames for simulation testing
 * without requiring physical drone playback or microphone permissions.
 */
export function synthesizeSimulationFrame(
  profile: SimulationProfile,
  timeOffsetSec: number,
  sampleRate = TARGET_SAMPLE_RATE,
  numSamples = WINDOW_SIZE_SAMPLES
): Float32Array {
  const frame = new Float32Array(numSamples);

  // Deterministic LCG PRNG seeded per time step so ambient noise evolves naturally
  let rngState = (Math.floor(timeOffsetSec * 1000) * 1664525 + 1013904223) >>> 0;
  const nextRand = (): number => {
    rngState = (Math.imul(1664525, rngState) + 1013904223) >>> 0;
    return (rngState / 4294967296) * 2 - 1;
  };

  if (profile === 'shahed') {
    // Shahed-136 / Limbach L550E two-stroke piston harmonic comb centered at 92.5 Hz up to 4 kHz
    const maxHarmonic = Math.floor(4000 / 92.5);
    let lpNoise = 0;

    for (let i = 0; i < numSamples; i += 1) {
      const t = timeOffsetSec + i / sampleRate;
      // Gentle flyby Doppler drift (~ +/- 2.2 Hz) and slight RPM wobble
      const dopplerFactor = 1 + 0.024 * Math.sin(2 * Math.PI * 0.08 * t);
      const f0 = 92.5 * dopplerFactor;

      let comb = 0;
      for (let k = 1; k <= maxHarmonic; k += 1) {
        const freq = k * f0;
        if (freq > 4000) break;
        // Two-stroke exhaust comb envelope: strong fundamental + 1/k^0.78 harmonic decay + mild air roll-off
        const env = (1 / Math.pow(k, 0.78)) * Math.exp(-freq / 3800);
        const phaseShift = k * 0.37;
        comb += env * Math.sin(2 * Math.PI * freq * t + phaseShift);
      }

      // 18 Hz prop-tip modulation + light ambient wind noise
      const propMod = 1 + 0.08 * Math.sin(2 * Math.PI * 18.5 * t);
      const white = nextRand();
      lpNoise = 0.85 * lpNoise + 0.15 * white;
      frame[i] = 0.28 * comb * propMod + 0.025 * lpNoise + 0.008 * white;
    }
    return frame;
  }

  if (profile === 'multirotor') {
    // Quadcopter 4-rotor blade passing frequencies (175-198 Hz) + 26 Hz blade flutter
    const rotorBaseFreqs = [174.0, 181.5, 189.0, 196.5];
    let lpNoise = 0;

    for (let i = 0; i < numSamples; i += 1) {
      const t = timeOffsetSec + i / sampleRate;
      const flutter = 1 + 0.12 * Math.sin(2 * Math.PI * 26.0 * t);

      let sig = 0;
      for (let r = 0; r < rotorBaseFreqs.length; r += 1) {
        const fBase = rotorBaseFreqs[r] * (1 + 0.01 * Math.sin(2 * Math.PI * (0.2 + r * 0.07) * t));
        for (let h = 1; h <= 10; h += 1) {
          const freq = fBase * h;
          if (freq > 3600) break;
          sig += (0.25 / Math.pow(h, 0.9)) * Math.sin(2 * Math.PI * freq * t + r * 1.1);
        }
      }

      const white = nextRand();
      lpNoise = 0.7 * lpNoise + 0.3 * white;
      frame[i] = 0.24 * sig * flutter + 0.035 * lpNoise + 0.012 * white;
    }
    return frame;
  }

  // Environmental background: low-frequency wind rumble + gentle broadband air noise
  let brown = 0;
  let pink = 0;
  for (let i = 0; i < numSamples; i += 1) {
    const t = timeOffsetSec + i / sampleRate;
    const white = nextRand();
    brown = 0.97 * brown + 0.03 * white;
    pink = 0.82 * pink + 0.18 * white;
    const gust = 0.6 + 0.4 * Math.sin(2 * Math.PI * 0.35 * t);
    frame[i] = 0.12 * brown * gust + 0.04 * pink + 0.015 * white;
  }
  return frame;
}

export class StreamManager {
  private readonly callbacks: StreamManagerCallbacks;

  private audioContext: AudioContext | null = null;
  private mediaStream: MediaStream | null = null;
  private sourceNode: MediaStreamAudioSourceNode | null = null;
  private workletNode: AudioWorkletNode | null = null;
  private muteGainNode: GainNode | null = null;

  private simTimer: ReturnType<typeof setInterval> | null = null;
  private simTimeSec = 0;

  private state: AudioCaptureState = 'idle';
  private activeSampleRate: number = TARGET_SAMPLE_RATE;

  constructor(callbacks: StreamManagerCallbacks) {
    this.callbacks = callbacks;
  }

  public getState(): AudioCaptureState {
    return this.state;
  }

  public getSampleRate(): number {
    return this.activeSampleRate;
  }

  public async startMicrophone(): Promise<void> {
    this.stop();

    if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
      const msg = 'Microphone capture unavailable in this browser context.';
      this.callbacks.onError?.(msg);
      throw new Error(msg);
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          channelCount: 1,
          sampleRate: TARGET_SAMPLE_RATE,
        },
        video: false,
      });

      const ctx = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
      if (ctx.state === 'suspended') {
        await ctx.resume();
      }

      await ctx.audioWorklet.addModule(workletUrl);

      const source = ctx.createMediaStreamSource(stream);
      const worklet = new AudioWorkletNode(ctx, 'panotti-audio-processor', {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        channelCount: 1,
        processorOptions: {
          inputSampleRate: ctx.sampleRate,
          targetSampleRate: TARGET_SAMPLE_RATE,
        },
      });

      // Keep the graph pulling without playing mic audio back to speakers
      const muteGain = ctx.createGain();
      muteGain.gain.value = 0;

      worklet.port.onmessage = (event: MessageEvent<{ type?: string; samples?: Float32Array }>) => {
        if (event.data?.type === 'audio_frame' && event.data.samples instanceof Float32Array) {
          this.callbacks.onFrame(event.data.samples);
        }
      };
      worklet.onprocessorerror = () => {
        this.callbacks.onError?.('AudioWorklet processor encountered a runtime error.');
      };

      source.connect(worklet);
      worklet.connect(muteGain);
      muteGain.connect(ctx.destination);

      this.mediaStream = stream;
      this.audioContext = ctx;
      this.sourceNode = source;
      this.workletNode = worklet;
      this.muteGainNode = muteGain;
      this.activeSampleRate = ctx.sampleRate;
      this.setState('microphone', ctx.sampleRate);
    } catch (err) {
      this.stop();
      const message = err instanceof Error ? err.message : 'Failed to start microphone capture';
      this.callbacks.onError?.(message);
      throw err;
    }
  }

  public startSimulation(profile: SimulationProfile): void {
    this.stop();
    this.simTimeSec = 0;
    this.activeSampleRate = TARGET_SAMPLE_RATE;
    this.setState('simulation', TARGET_SAMPLE_RATE);

    const hopSec = HOP_SIZE_SAMPLES / TARGET_SAMPLE_RATE; // 0.25 s

    const emitNext = () => {
      if (this.state !== 'simulation') return;
      const frame = synthesizeSimulationFrame(profile, this.simTimeSec, TARGET_SAMPLE_RATE);
      this.simTimeSec += hopSec;
      this.callbacks.onFrame(frame);
    };

    emitNext();
    this.simTimer = setInterval(emitNext, 250);
  }

  public stop(): void {
    if (this.simTimer !== null) {
      clearInterval(this.simTimer);
      this.simTimer = null;
    }

    if (this.workletNode) {
      this.workletNode.port.onmessage = null;
      this.workletNode.disconnect();
      this.workletNode = null;
    }

    if (this.sourceNode) {
      this.sourceNode.disconnect();
      this.sourceNode = null;
    }

    if (this.muteGainNode) {
      this.muteGainNode.disconnect();
      this.muteGainNode = null;
    }

    if (this.mediaStream) {
      for (const track of this.mediaStream.getTracks()) {
        track.stop();
      }
      this.mediaStream = null;
    }

    if (this.audioContext) {
      void this.audioContext.close().catch(() => {});
      this.audioContext = null;
    }

    this.activeSampleRate = TARGET_SAMPLE_RATE;
    if (this.state !== 'idle') {
      this.setState('idle', TARGET_SAMPLE_RATE);
    }
  }

  private setState(state: AudioCaptureState, sampleRate: number): void {
    this.state = state;
    this.callbacks.onStateChange?.(state, sampleRate);
  }
}
