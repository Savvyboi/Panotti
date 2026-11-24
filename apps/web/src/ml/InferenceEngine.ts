import {
  UAV_CLASSES,
  type AudioClass,
  type EngineStatus,
  type ExecutionProviderUsed,
  type InferenceResult,
} from '../types';

export const MODEL_URL = '/models/uav_detector_int8.onnx';
export const ORT_WASM_PATHS = '/ort/';
export const SPECTROGRAM_SHAPE = [1, 1, 128, 100] as const;
export const SPECTROGRAM_LENGTH = 128 * 100;

export interface OrtTensorLike {
  readonly data: Float32Array | ArrayLike<number>;
  readonly dims: readonly number[];
  dispose?: () => void;
}

export interface OrtSessionLike {
  run(feeds: Record<string, OrtTensorLike>): Promise<Record<string, OrtTensorLike>>;
  release?: () => Promise<void>;
}

export interface OrtEnvWasmLike {
  wasmPaths?: string | Record<string, string>;
  numThreads?: number;
  simd?: boolean;
}

export interface OrtLike {
  env: {
    wasm: OrtEnvWasmLike;
  };
  Tensor: new (type: 'float32', data: Float32Array, dims: readonly number[]) => OrtTensorLike;
  InferenceSession: {
    create(
      uriOrBuffer: string | Uint8Array,
      options?: { executionProviders?: string[] }
    ): Promise<OrtSessionLike>;
  };
}

export interface InferenceEngineOptions {
  modelUrl?: string;
  wasmPaths?: string;
  ortModule?: OrtLike;
  fetchImpl?: typeof fetch;
  hasWebGpu?: () => boolean;
  isCrossOriginIsolated?: () => boolean;
  hardwareConcurrency?: number;
}

export interface EngineDiagnostics {
  status: EngineStatus;
  provider: ExecutionProviderUsed | null;
  crossOriginIsolated: boolean;
  configuredThreads: number;
  wasmPaths: string;
  message: string;
}

/**
 * Numerically stable softmax over raw class logits.
 */
export function softmax(logits: Float32Array | readonly number[]): number[] {
  const len = logits.length;
  if (len === 0) return [];

  let maxVal = -Infinity;
  for (let i = 0; i < len; i += 1) {
    const v = logits[i];
    if (v > maxVal) maxVal = v;
  }

  const exps = new Array<number>(len);
  let sum = 0;
  for (let i = 0; i < len; i += 1) {
    const e = Math.exp(logits[i] - maxVal);
    exps[i] = e;
    sum += e;
  }

  if (sum <= 0 || !Number.isFinite(sum)) {
    const uniform = 1 / len;
    return new Array<number>(len).fill(uniform);
  }

  for (let i = 0; i < len; i += 1) {
    exps[i] /= sum;
  }
  return exps;
}

export class InferenceEngine {
  private readonly modelUrl: string;
  private readonly wasmPaths: string;
  private readonly injectedOrt?: OrtLike;
  private readonly fetchImpl?: typeof fetch;
  private readonly hasWebGpuFn: () => boolean;
  private readonly isCrossOriginIsolatedFn: () => boolean;
  private readonly hardwareConcurrency: number;

  private ort: OrtLike | null = null;
  private session: OrtSessionLike | null = null;
  private status: EngineStatus = 'uninitialized';
  private provider: ExecutionProviderUsed | null = null;
  private configuredThreads = 1;
  private message = 'Inference engine not initialized';
  private initPromise: Promise<boolean> | null = null;

  constructor(options: InferenceEngineOptions = {}) {
    this.modelUrl = options.modelUrl ?? MODEL_URL;
    this.wasmPaths = options.wasmPaths ?? ORT_WASM_PATHS;
    this.injectedOrt = options.ortModule;
    this.fetchImpl =
      options.fetchImpl ??
      (options.ortModule
        ? undefined
        : typeof fetch === 'function'
          ? fetch.bind(globalThis)
          : undefined);
    this.hasWebGpuFn =
      options.hasWebGpu ??
      (() => typeof navigator !== 'undefined' && 'gpu' in navigator);
    this.isCrossOriginIsolatedFn =
      options.isCrossOriginIsolated ??
      (() =>
        typeof globalThis !== 'undefined' &&
        Boolean((globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated));
    this.hardwareConcurrency =
      options.hardwareConcurrency ??
      (typeof navigator !== 'undefined' && typeof navigator.hardwareConcurrency === 'number'
        ? navigator.hardwareConcurrency
        : 4);
  }

  public getDiagnostics(): EngineDiagnostics {
    return {
      status: this.status,
      provider: this.provider,
      crossOriginIsolated: this.isCrossOriginIsolatedFn(),
      configuredThreads: this.configuredThreads,
      wasmPaths: this.wasmPaths,
      message: this.message,
    };
  }

  public isReady(): boolean {
    return this.status === 'ready' && this.session !== null;
  }

  public async init(): Promise<boolean> {
    if (this.status === 'ready' && this.session) return true;
    if (this.initPromise) return this.initPromise;

    this.initPromise = this.performInit();
    try {
      return await this.initPromise;
    } finally {
      this.initPromise = null;
    }
  }

  private async loadModelSource(): Promise<{ ok: true; source: string | Uint8Array } | { ok: false; reason: string }> {
    if (!this.fetchImpl) {
      return { ok: true, source: this.modelUrl };
    }

    try {
      const response = await this.fetchImpl(this.modelUrl);
      if (!response.ok) {
        return {
          ok: false,
          reason: `Model file unavailable at ${this.modelUrl} (HTTP ${response.status}). Run ML export or deploy uav_detector_int8.onnx.`,
        };
      }

      const contentType = response.headers?.get('content-type')?.toLowerCase() ?? '';
      if (contentType.includes('text/html')) {
        return {
          ok: false,
          reason: `Model not found at ${this.modelUrl} (received HTML fallback). Waiting for ML pipeline export.`,
        };
      }

      const buffer = await response.arrayBuffer();
      if (buffer.byteLength < 32) {
        return {
          ok: false,
          reason: `Model at ${this.modelUrl} is empty or invalid (${buffer.byteLength} bytes).`,
        };
      }

      return { ok: true, source: new Uint8Array(buffer) };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      return {
        ok: false,
        reason: `Failed to fetch ONNX model at ${this.modelUrl}: ${msg}`,
      };
    }
  }

  private async performInit(): Promise<boolean> {
    this.status = 'loading';
    this.provider = null;
    this.message = 'Initializing ONNX Runtime Web...';

    try {
      const ort = this.injectedOrt ?? ((await import('onnxruntime-web')) as unknown as OrtLike);
      this.ort = ort;

      // Always serve WASM/MJS binaries locally from /ort/ so COEP never blocks CDN fetches
      ort.env.wasm.wasmPaths = this.wasmPaths;
      ort.env.wasm.simd = true;

      const isolated = this.isCrossOriginIsolatedFn();
      this.configuredThreads = isolated
        ? Math.min(4, Math.max(2, Math.floor(this.hardwareConcurrency)))
        : 1;
      ort.env.wasm.numThreads = this.configuredThreads;

      const modelLoad = await this.loadModelSource();
      if (!modelLoad.ok) {
        this.status = 'unavailable';
        this.message = modelLoad.reason;
        return false;
      }

      const modelSource = modelLoad.source;
      let fallbackNote = '';

      if (this.hasWebGpuFn()) {
        try {
          this.session = await ort.InferenceSession.create(modelSource, {
            executionProviders: ['webgpu'],
          });
          this.provider = 'webgpu';
          this.status = 'ready';
          this.message = 'WebGPU inference session active';
          return true;
        } catch (gpuErr) {
          const reason = gpuErr instanceof Error ? gpuErr.message : String(gpuErr);
          fallbackNote = `WebGPU unavailable (${reason}); `;
        }
      }

      this.session = await ort.InferenceSession.create(modelSource, {
        executionProviders: ['wasm'],
      });
      this.provider = 'wasm';
      this.status = 'ready';
      this.message = `${fallbackNote}WASM-SIMD active (${this.configuredThreads} thread${this.configuredThreads > 1 ? 's' : ''})`;
      return true;
    } catch (err) {
      const reason = err instanceof Error ? err.message : String(err);
      const looksMissing =
        reason.includes('404') ||
        reason.toLowerCase().includes('protobuf') ||
        reason.toLowerCase().includes('fetch');
      this.status = looksMissing ? 'unavailable' : 'error';
      this.session = null;
      this.provider = null;
      this.message = `Inference init failed: ${reason}`;
      return false;
    }
  }

  public async run(spectrogram: Float32Array): Promise<InferenceResult> {
    if (spectrogram.length !== SPECTROGRAM_LENGTH) {
      throw new RangeError(
        `Expected spectrogram of length ${SPECTROGRAM_LENGTH}, received ${spectrogram.length}`
      );
    }
    if (!this.ort || !this.session || this.status !== 'ready' || !this.provider) {
      throw new Error(`InferenceEngine is not ready (status: ${this.status})`);
    }

    const start = typeof performance !== 'undefined' ? performance.now() : Date.now();
    const inputTensor = new this.ort.Tensor('float32', spectrogram, SPECTROGRAM_SHAPE);
    let outputs: Record<string, OrtTensorLike> | null = null;

    try {
      outputs = await this.session.run({ input_spectrogram: inputTensor });
      const logitsTensor = outputs.class_logits;
      if (!logitsTensor || logitsTensor.data.length < UAV_CLASSES.length) {
        throw new Error('Model output "class_logits" missing or invalid shape');
      }

      const logits = new Array<number>(UAV_CLASSES.length);
      for (let i = 0; i < UAV_CLASSES.length; i += 1) {
        logits[i] = Number(logitsTensor.data[i]);
      }

      const probs = softmax(logits);
      const probabilities = {} as Record<AudioClass, number>;

      let topIndex = 0;
      let topProb = -1;
      for (let i = 0; i < UAV_CLASSES.length; i += 1) {
        const cls = UAV_CLASSES[i];
        probabilities[cls] = probs[i];
        if (probs[i] > topProb) {
          topProb = probs[i];
          topIndex = i;
        }
      }

      const end = typeof performance !== 'undefined' ? performance.now() : Date.now();
      return {
        topClass: UAV_CLASSES[topIndex],
        confidence: topProb,
        probabilities,
        logits,
        provider: this.provider,
        inferenceMs: Math.max(0, Number((end - start).toFixed(2))),
      };
    } finally {
      inputTensor.dispose?.();
      if (outputs) {
        for (const tensor of Object.values(outputs)) {
          tensor.dispose?.();
        }
      }
    }
  }

  public async dispose(): Promise<void> {
    if (this.session?.release) {
      await this.session.release().catch(() => {});
    }
    this.session = null;
    this.provider = null;
    this.status = 'uninitialized';
    this.message = 'Disposed';
  }
}
