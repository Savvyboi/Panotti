import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, statSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..');
const webRoot = path.join(repoRoot, 'apps', 'web');
const pythonExe = path.join(
  repoRoot,
  'pipelines',
  'dataset_builder',
  '.venv',
  process.platform === 'win32' ? 'Scripts' : 'bin',
  process.platform === 'win32' ? 'python.exe' : 'python'
);
const modelPath = path.join(webRoot, 'public', 'models', 'uav_detector_int8.onnx');

const UAV_CLASSES = [
  'Shahed_Piston_UAV',
  'Multirotor_UAV',
  'FixedWing_Electric_UAV',
  'Ground_Combustion_Noise',
  'Environmental_Background',
];

function softmax(logits) {
  const maxVal = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - maxVal));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

async function main() {
  const modelBytes = statSync(modelPath).size;
  if (modelBytes >= 8 * 1024 * 1024) {
    throw new Error(`Model too large: ${modelBytes} bytes (>= 8 MB)`);
  }

  const tmpDir = mkdtempSync(path.join(os.tmpdir(), 'panotti-parity-'));
  const jsonPath = path.join(tmpDir, 'py_ref.json');

  try {
    execFileSync(pythonExe, [path.join(__dirname, 'verify_parity.py'), jsonPath], {
      stdio: 'inherit',
    });
    const pyRef = JSON.parse(readFileSync(jsonPath, 'utf-8'));

    const { createServer } = await import(
      pathToFileURL(path.join(webRoot, 'node_modules', 'vite', 'dist', 'node', 'index.js')).href
    );
    const vite = await createServer({
      root: webRoot,
      logLevel: 'silent',
      server: { middlewareMode: true },
      appType: 'custom',
    });

    try {
      const melMod = await vite.ssrLoadModule('/src/audio/MelSpectrogramExtractor.ts');
      const extractor = new melMod.MelSpectrogramExtractor();

      const pcm = Float32Array.from(pyRef.pcm);
      const tsSpec = extractor.extract(pcm);
      const pySpec = Float32Array.from(pyRef.spectrogram);

      if (tsSpec.length !== 12800 || tsSpec.length !== pySpec.length) {
        throw new Error(`Length mismatch: ts=${tsSpec.length}, py=${pySpec.length}`);
      }

      let maxSpecDiff = 0;
      for (let i = 0; i < tsSpec.length; i += 1) {
        const d = Math.abs(tsSpec[i] - pySpec[i]);
        if (d > maxSpecDiff) maxSpecDiff = d;
      }
      if (maxSpecDiff >= 1e-3) {
        throw new Error(`Python vs TS spectrogram parity exceeded 1e-3: ${maxSpecDiff}`);
      }

      // Run ONNX Runtime Web (WASM) in Node on TS-extracted spectrogram
      const ort = await import(
        pathToFileURL(path.join(webRoot, 'node_modules', 'onnxruntime-web', 'dist', 'ort.wasm.min.mjs')).href
      );
      ort.env.wasm.wasmPaths = pathToFileURL(path.join(webRoot, 'public', 'ort') + path.sep).href;
      ort.env.wasm.numThreads = 1;

      const modelBytesBuf = new Uint8Array(readFileSync(modelPath));
      const session = await ort.InferenceSession.create(modelBytesBuf, {
        executionProviders: ['wasm'],
      });

      const tensor = new ort.Tensor('float32', tsSpec, [1, 1, 128, 100]);
      const outputs = await session.run({ input_spectrogram: tensor });
      const tsLogits = Array.from(outputs.class_logits.data, Number);
      const tsProbs = softmax(tsLogits);
      const topIdx = tsProbs.indexOf(Math.max(...tsProbs));
      const topClass = UAV_CLASSES[topIdx];
      const topConf = tsProbs[topIdx];

      let maxLogitDiff = 0;
      for (let i = 0; i < tsLogits.length; i += 1) {
        const d = Math.abs(tsLogits[i] - pyRef.logits[i]);
        if (d > maxLogitDiff) maxLogitDiff = d;
      }

      // Also verify StreamManager simulation frames classify accurately
      const simMod = await vite.ssrLoadModule('/src/audio/StreamManager.ts');
      const simChecks = {};
      for (const [profile, expectedClass] of [
        ['shahed', 'Shahed_Piston_UAV'],
        ['multirotor', 'Multirotor_UAV'],
        ['background', 'Environmental_Background'],
      ]) {
        const simPcm = simMod.synthesizeSimulationFrame(profile, 0.25);
        const simSpec = extractor.extract(simPcm);
        const simTensor = new ort.Tensor('float32', simSpec, [1, 1, 128, 100]);
        const simOut = await session.run({ input_spectrogram: simTensor });
        const simProbs = softmax(Array.from(simOut.class_logits.data, Number));
        const simTopIdx = simProbs.indexOf(Math.max(...simProbs));
        simChecks[profile] = {
          expected: expectedClass,
          predicted: UAV_CLASSES[simTopIdx],
          confidence: Number(simProbs[simTopIdx].toFixed(6)),
        };
        simTensor.dispose?.();
        if (UAV_CLASSES[simTopIdx] !== expectedClass || simProbs[simTopIdx] < 0.8) {
          throw new Error(
            `Simulation profile "${profile}" mismatch: got ${UAV_CLASSES[simTopIdx]} (${simProbs[simTopIdx]})`
          );
        }
      }

      tensor.dispose?.();
      await session.release?.();

      if (topClass !== 'Shahed_Piston_UAV' || topConf < 0.8) {
        throw new Error(`Expected Shahed_Piston_UAV >= 0.80, got ${topClass} (${topConf})`);
      }

      console.log(
        JSON.stringify(
          {
            status: 'PASS',
            modelBytes,
            modelMB: Number((modelBytes / (1024 * 1024)).toFixed(3)),
            specMaxAbsDiff: maxSpecDiff,
            ortLogitMaxAbsDiff: maxLogitDiff,
            deterministicComb92_5Hz: {
              topClass,
              confidence: Number(topConf.toFixed(6)),
              pythonTopClass: pyRef.topClass,
              pythonConfidence: Number(pyRef.confidence.toFixed(6)),
            },
            simulationProfiles: simChecks,
          },
          null,
          2
        )
      );
    } finally {
      await vite.close();
    }
  } finally {
    rmSync(tmpDir, { recursive: true, force: true });
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
