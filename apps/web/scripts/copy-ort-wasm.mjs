import { copyFileSync, existsSync, mkdirSync, readdirSync, rmSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const webRoot = path.resolve(__dirname, '..');

const ortDistDir = path.join(webRoot, 'node_modules', 'onnxruntime-web', 'dist');
const publicOrtDir = path.join(webRoot, 'public', 'ort');
const publicModelsDir = path.join(webRoot, 'public', 'models');

const CLOUDFLARE_PAGES_MAX_BYTES = 24 * 1024 * 1024;
const pagesSafe =
  process.env.PANOTTI_PAGES_SAFE === '1' || process.argv.includes('--pages-safe');

// Ensure public/models directory exists without modifying any existing model files.
mkdirSync(publicModelsDir, { recursive: true });
rmSync(publicOrtDir, { recursive: true, force: true });
mkdirSync(publicOrtDir, { recursive: true });

if (!existsSync(ortDistDir)) {
  console.warn(`[copy-ort-wasm] Skipping: ${ortDistDir} not found yet.`);
  process.exit(0);
}

const entries = readdirSync(ortDistDir, { withFileTypes: true });
let copied = 0;
let skippedLarge = 0;

for (const entry of entries) {
  if (!entry.isFile()) continue;
  const isOrtWasmRuntime =
    entry.name.startsWith('ort-wasm-') &&
    (entry.name.endsWith('.wasm') || entry.name.endsWith('.mjs'));
  if (!isOrtWasmRuntime) continue;

  const src = path.join(ortDistDir, entry.name);
  if (pagesSafe && entry.name.endsWith('.wasm') && statSync(src).size > CLOUDFLARE_PAGES_MAX_BYTES) {
    skippedLarge += 1;
    continue;
  }

  const dst = path.join(publicOrtDir, entry.name);
  copyFileSync(src, dst);
  copied += 1;
}

console.log(
  `[copy-ort-wasm] Copied ${copied} ORT WASM runtime assets to ${publicOrtDir}` +
    (skippedLarge > 0 ? ` (skipped ${skippedLarge} >24 MiB file(s) for Cloudflare Pages)` : '')
);
