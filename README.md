# Panotti — Acoustic UAV Detector & Distributed Mesh

Full-stack distributed acoustic detection and tracking system targeting multirotor consumer drones (e.g., DJI Mavic series) and low-frequency two-stroke piston-engine UAVs (e.g., Shahed-136 / Geran-2 with MD-550 / Limbach L550E engines).

## Monorepo Architecture

```text
Panotti/
├── .github/workflows/           # CI/CD workflows (ML export & Cloudflare deploy)
├── apps/web/                    # React + Vite PWA (AudioWorklet + ONNX Runtime Web + Leaflet)
├── services/mesh-hub/           # Cloudflare Workers + Durable Objects (H3 Res 6 + Kalman + BearingFusion)
├── pipelines/dataset_builder/   # Python OSINT ingestion, filtering, physics augmentation, training & ONNX export
└── scripts/                     # Cross-stack DSP & ONNX parity verification scripts
```

## Quick Start

### 1. Python ML Pipeline (`pipelines/dataset_builder`)
```bash
cd pipelines/dataset_builder
uv venv .venv
# On Windows: .venv\Scripts\activate  |  On Linux/macOS: source .venv/bin/activate
uv pip install -r requirements.txt
pytest -v
python train.py --synthetic --epochs 20 --num-samples-per-class 300
python export_onnx.py
```

For real OSINT scraping & cleaning (requires `yt-dlp`, `ffmpeg`, and optional `requirements-optional.txt` for Demucs/YAMNet):
```bash
python scrape_osint.py --queries "shahed 136 sound" "moped drone sound"
python clean_and_filter.py --input-dir data/raw --output-dir data/clean
python physics_augment.py --input-dir data/clean --output-dir data/augmented
```

### 2. Cloudflare Mesh Hub (`services/mesh-hub`)
```bash
cd services/mesh-hub
npm install
npm test
npx wrangler dev --port 8787
```

### 3. Web Client PWA (`apps/web`)
```bash
cd apps/web
npm install
npm test
npm run build
npm run preview -- --port 4173 --host 127.0.0.1
```

## Shared DSP & Model Contract

Both `pipelines/dataset_builder/feature_extraction.py` and `apps/web/src/audio/MelSpectrogramExtractor.ts` implement the identical Log-Mel pipeline:
- **Sample Rate**: `16000 Hz`, mono float32, 1-second frame (`16000` samples)
- **STFT**: `N_FFT = 512`, `HOP_LENGTH = 160` (`100` frames, right-padded to `16352`), periodic Hann window
- **Mel Filterbank**: `128` HTK Mel bins over `50 Hz – 4000 Hz` with low-frequency center interpolation fallback and L1 row normalization
- **Normalization**: `ln(max(energy, 1e-6))` followed by per-spectrogram z-score normalization
- **Tensor Shape**: `[1, 1, 128, 100]` (`input_spectrogram` -> `class_logits`)
- **Classes**: `Shahed_Piston_UAV`, `Multirotor_UAV`, `FixedWing_Electric_UAV`, `Ground_Combustion_Noise`, `Environmental_Background`
