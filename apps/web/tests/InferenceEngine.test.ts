import { describe, expect, it, vi } from 'vitest';
import {
  InferenceEngine,
  softmax,
  SPECTROGRAM_LENGTH,
  type OrtLike,
} from '../src/ml/InferenceEngine';

function createMockOrt(options?: {
  failWebGpu?: boolean;
  logits?: number[];
}): {
  ort: OrtLike;
  createCalls: Array<string[] | undefined>;
  runCalls: Array<{ inputName: string; dims: readonly number[] }>;
} {
  const createCalls: Array<string[] | undefined> = [];
  const runCalls: Array<{ inputName: string; dims: readonly number[] }> = [];
  const logits = options?.logits ?? [4.0, 1.0, 0.2, -1.0, -2.0];

  const ort: OrtLike = {
    env: {
      wasm: {},
    },
    Tensor: class MockTensor {
      public readonly data: Float32Array;
      public readonly dims: readonly number[];
      constructor(_type: 'float32', data: Float32Array, dims: readonly number[]) {
        this.data = data;
        this.dims = dims;
      }
      dispose() {}
    },
    InferenceSession: {
      async create(_src, createOpts) {
        const eps = createOpts?.executionProviders;
        createCalls.push(eps);
        if (options?.failWebGpu && eps?.includes('webgpu')) {
          throw new Error('WebGPU adapter unavailable');
        }
        return {
          async run(feeds) {
            const input = feeds.input_spectrogram;
            runCalls.push({
              inputName: 'input_spectrogram',
              dims: input.dims,
            });
            return {
              class_logits: {
                data: new Float32Array(logits),
                dims: [1, 5],
                dispose: () => {},
              },
            };
          },
          async release() {},
        };
      },
    },
  };

  return { ort, createCalls, runCalls };
}

describe('InferenceEngine & softmax', () => {
  it('computes numerically stable softmax summing to 1.0', () => {
    const probs = softmax([1000, 1002, 999, 998, 990]);
    expect(probs).toHaveLength(5);
    const total = probs.reduce((a, b) => a + b, 0);
    expect(total).toBeCloseTo(1.0, 10);
    expect(probs[1]).toBeGreaterThan(probs[0]);
    expect(probs[1]).toBeGreaterThan(0.8);
  });

  it('sets wasmPaths=/ort/ and gates numThreads on crossOriginIsolated', async () => {
    const { ort: ortUnisolated } = createMockOrt();
    const unisolated = new InferenceEngine({
      ortModule: ortUnisolated,
      hasWebGpu: () => false,
      isCrossOriginIsolated: () => false,
      hardwareConcurrency: 8,
    });
    expect(await unisolated.init()).toBe(true);
    expect(ortUnisolated.env.wasm.wasmPaths).toBe('/ort/');
    expect(ortUnisolated.env.wasm.numThreads).toBe(1);

    const { ort: ortIsolated } = createMockOrt();
    const isolated = new InferenceEngine({
      ortModule: ortIsolated,
      hasWebGpu: () => false,
      isCrossOriginIsolated: () => true,
      hardwareConcurrency: 8,
    });
    expect(await isolated.init()).toBe(true);
    expect(ortIsolated.env.wasm.wasmPaths).toBe('/ort/');
    expect(ortIsolated.env.wasm.numThreads).toBe(4);
  });

  it('tries WebGPU first and falls back cleanly to WASM on failure', async () => {
    const { ort, createCalls, runCalls } = createMockOrt({
      failWebGpu: true,
      logits: [5.0, 1.0, 0.0, -1.0, -2.0],
    });

    const engine = new InferenceEngine({
      ortModule: ort,
      hasWebGpu: () => true,
      isCrossOriginIsolated: () => true,
    });

    const ok = await engine.init();
    expect(ok).toBe(true);
    expect(createCalls).toEqual([['webgpu'], ['wasm']]);

    const diag = engine.getDiagnostics();
    expect(diag.status).toBe('ready');
    expect(diag.provider).toBe('wasm');

    const res = await engine.run(new Float32Array(SPECTROGRAM_LENGTH));
    expect(runCalls[0]).toEqual({ inputName: 'input_spectrogram', dims: [1, 1, 128, 100] });
    expect(res.topClass).toBe('Shahed_Piston_UAV');
    expect(res.confidence).toBeGreaterThan(0.95);
    expect(res.provider).toBe('wasm');
  });

  it('uses WebGPU directly when available and healthy', async () => {
    const { ort, createCalls } = createMockOrt({
      failWebGpu: false,
      logits: [0.1, 4.5, 0.2, -1.0, -1.0],
    });

    const engine = new InferenceEngine({
      ortModule: ort,
      hasWebGpu: () => true,
      isCrossOriginIsolated: () => true,
    });

    expect(await engine.init()).toBe(true);
    expect(createCalls).toEqual([['webgpu']]);
    expect(engine.getDiagnostics().provider).toBe('webgpu');

    const res = await engine.run(new Float32Array(SPECTROGRAM_LENGTH));
    expect(res.topClass).toBe('Multirotor_UAV');
  });

  it('reports unavailable gracefully when model fetch returns 404 or HTML fallback', async () => {
    const { ort } = createMockOrt();
    const fetch404 = vi.fn(async () => new Response('Not Found', { status: 404 }));

    const engine404 = new InferenceEngine({
      ortModule: ort,
      fetchImpl: fetch404 as unknown as typeof fetch,
    });
    expect(await engine404.init()).toBe(false);
    expect(engine404.getDiagnostics().status).toBe('unavailable');

    const fetchHtml = vi.fn(
      async () =>
        new Response('<!doctype html><html></html>', {
          status: 200,
          headers: { 'content-type': 'text/html' },
        })
    );
    const engineHtml = new InferenceEngine({
      ortModule: ort,
      fetchImpl: fetchHtml as unknown as typeof fetch,
    });
    expect(await engineHtml.init()).toBe(false);
    expect(engineHtml.getDiagnostics().status).toBe('unavailable');
  });
});
