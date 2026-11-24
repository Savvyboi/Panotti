export const TARGET_SAMPLE_RATE = 16000;
export const WINDOW_SIZE_SAMPLES = 16000;
export const HOP_SIZE_SAMPLES = 4000;

/**
 * Pure circular ring-buffer and causal linear resampler (with anti-alias biquad LPF when downsampling).
 * Collects incoming audio chunks at `inputSampleRate` and emits contiguous 16,000-sample frames
 * every 4,000 output samples (250 ms hop at 16 kHz) once the initial 1-second buffer is filled.
 */
export class RingBufferResampler {
  public readonly inputSampleRate: number;
  public readonly targetSampleRate: number;
  public readonly windowSize: number;
  public readonly hopSize: number;

  private readonly ringBuffer: Float32Array;
  private readonly ratio: number;
  private readonly needsResample: boolean;
  private readonly needsLowpass: boolean;

  private writePos = 0;
  private totalWritten = 0;
  private samplesSinceEmit = 0;

  // Resampler interpolation state
  private initialized = false;
  private prevSample = 0;
  private nextOffset = 0;

  // 2nd-order Butterworth anti-aliasing biquad state (used when inputSampleRate > targetSampleRate)
  private b0 = 1;
  private b1 = 0;
  private b2 = 0;
  private a1 = 0;
  private a2 = 0;
  private z1 = 0;
  private z2 = 0;

  constructor(
    inputSampleRate: number,
    targetSampleRate = TARGET_SAMPLE_RATE,
    windowSize = WINDOW_SIZE_SAMPLES,
    hopSize = HOP_SIZE_SAMPLES
  ) {
    if (!Number.isFinite(inputSampleRate) || inputSampleRate <= 0) {
      throw new RangeError(`Invalid inputSampleRate: ${inputSampleRate}`);
    }
    if (!Number.isFinite(targetSampleRate) || targetSampleRate <= 0) {
      throw new RangeError(`Invalid targetSampleRate: ${targetSampleRate}`);
    }
    if (windowSize <= 0 || hopSize <= 0) {
      throw new RangeError('windowSize and hopSize must be positive');
    }

    this.inputSampleRate = inputSampleRate;
    this.targetSampleRate = targetSampleRate;
    this.windowSize = windowSize;
    this.hopSize = hopSize;

    this.ringBuffer = new Float32Array(windowSize);
    this.ratio = inputSampleRate / targetSampleRate;
    this.needsResample = Math.abs(inputSampleRate - targetSampleRate) > 1e-6;
    this.needsLowpass = inputSampleRate > targetSampleRate * 1.01;

    if (this.needsLowpass) {
      this.configureLowpass(0.45 * targetSampleRate, inputSampleRate);
    }
  }

  private configureLowpass(cutoffHz: number, sampleRate: number): void {
    const omega = (2 * Math.PI * cutoffHz) / sampleRate;
    const sinW = Math.sin(omega);
    const cosW = Math.cos(omega);
    const alpha = sinW / Math.SQRT2;
    const a0 = 1 + alpha;

    this.b0 = (1 - cosW) / 2 / a0;
    this.b1 = (1 - cosW) / a0;
    this.b2 = (1 - cosW) / 2 / a0;
    this.a1 = (-2 * cosW) / a0;
    this.a2 = (1 - alpha) / a0;
  }

  private filterSample(x: number): number {
    if (!this.needsLowpass) return x;
    const y = this.b0 * x + this.z1;
    this.z1 = this.b1 * x - this.a1 * y + this.z2;
    this.z2 = this.b2 * x - this.a2 * y;
    return y;
  }

  private writeTargetSample(sample: number, emitted: Float32Array[]): void {
    this.ringBuffer[this.writePos] = sample;
    this.writePos = (this.writePos + 1) % this.windowSize;
    this.totalWritten += 1;

    if (this.totalWritten < this.windowSize) {
      return;
    }

    if (this.totalWritten === this.windowSize) {
      this.samplesSinceEmit = 0;
      emitted.push(this.snapshotFrame());
      return;
    }

    this.samplesSinceEmit += 1;
    if (this.samplesSinceEmit >= this.hopSize) {
      this.samplesSinceEmit = 0;
      emitted.push(this.snapshotFrame());
    }
  }

  private snapshotFrame(): Float32Array {
    const frame = new Float32Array(this.windowSize);
    const tailLength = this.windowSize - this.writePos;
    frame.set(this.ringBuffer.subarray(this.writePos), 0);
    if (this.writePos > 0) {
      frame.set(this.ringBuffer.subarray(0, this.writePos), tailLength);
    }
    return frame;
  }

  /**
   * Push an incoming chunk of mono PCM samples at `inputSampleRate`.
   * Returns 0 or more 16,000-sample frames ready for spectrogram extraction.
   */
  public push(chunk: Float32Array): Float32Array[] {
    const emitted: Float32Array[] = [];
    if (chunk.length === 0) return emitted;

    if (!this.needsResample) {
      for (let i = 0; i < chunk.length; i += 1) {
        this.writeTargetSample(chunk[i], emitted);
      }
      return emitted;
    }

    for (let i = 0; i < chunk.length; i += 1) {
      const filtered = this.filterSample(chunk[i]);

      if (!this.initialized) {
        this.writeTargetSample(filtered, emitted);
        this.prevSample = filtered;
        this.initialized = true;
        this.nextOffset = this.ratio;
        continue;
      }

      while (this.nextOffset <= 1) {
        const interp = this.prevSample + this.nextOffset * (filtered - this.prevSample);
        this.writeTargetSample(interp, emitted);
        this.nextOffset += this.ratio;
      }

      this.prevSample = filtered;
      this.nextOffset -= 1;
    }

    return emitted;
  }

  public getTotalOutputSamples(): number {
    return this.totalWritten;
  }

  public reset(): void {
    this.ringBuffer.fill(0);
    this.writePos = 0;
    this.totalWritten = 0;
    this.samplesSinceEmit = 0;
    this.initialized = false;
    this.prevSample = 0;
    this.nextOffset = 0;
    this.z1 = 0;
    this.z2 = 0;
  }
}

// AudioWorkletProcessor global declarations (only defined inside AudioWorkletGlobalScope)
declare abstract class AudioWorkletProcessor {
  readonly port: MessagePort;
  constructor(options?: AudioWorkletNodeOptions);
  abstract process(
    inputs: Float32Array[][],
    outputs: Float32Array[][],
    parameters: Record<string, Float32Array>
  ): boolean;
}

declare function registerProcessor(
  name: string,
  processorCtor: new (options?: AudioWorkletNodeOptions) => AudioWorkletProcessor
): void;

declare const sampleRate: number | undefined;

if (typeof AudioWorkletProcessor !== 'undefined' && typeof registerProcessor === 'function') {
  class PanottiAudioProcessor extends AudioWorkletProcessor {
    private readonly resampler: RingBufferResampler;

    constructor(options?: AudioWorkletNodeOptions) {
      super(options);
      const opts = (options?.processorOptions ?? {}) as {
        inputSampleRate?: number;
        targetSampleRate?: number;
      };
      const inputRate =
        opts.inputSampleRate ?? (typeof sampleRate === 'number' ? sampleRate : TARGET_SAMPLE_RATE);
      const targetRate = opts.targetSampleRate ?? TARGET_SAMPLE_RATE;
      this.resampler = new RingBufferResampler(
        inputRate,
        targetRate,
        WINDOW_SIZE_SAMPLES,
        HOP_SIZE_SAMPLES
      );

      this.port.onmessage = (event: MessageEvent<{ type?: string }>) => {
        if (event.data?.type === 'reset') {
          this.resampler.reset();
        }
      };
    }

    process(inputs: Float32Array[][]): boolean {
      const channel = inputs[0]?.[0];
      if (!channel || channel.length === 0) {
        return true;
      }

      const frames = this.resampler.push(channel);
      for (const frame of frames) {
        this.port.postMessage({ type: 'audio_frame', samples: frame }, [frame.buffer]);
      }
      return true;
    }
  }

  registerProcessor('panotti-audio-processor', PanottiAudioProcessor);
}
