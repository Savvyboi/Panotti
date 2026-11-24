import { useEffect, useRef } from 'react';
import { DSP_CONFIG, hzToMel } from '../audio/MelSpectrogramExtractor';
import type { AudioClass } from '../types';

export interface SpectrogramCanvasProps {
  spectrogram: Float32Array | null;
  peakFreqHz: number;
  activeClass: AudioClass | null;
  confidence: number;
}

const MEL_BINS = DSP_CONFIG.nMels; // 128
const FRAMES = DSP_CONFIG.numFrames; // 100

// Precompute 256-entry tactical thermal LUT (R, G, B)
const COLOR_LUT = (() => {
  const lut = new Uint8ClampedArray(256 * 3);
  const stops: Array<[number, [number, number, number]]> = [
    [0.0, [7, 11, 17]],
    [0.2, [15, 36, 68]],
    [0.42, [14, 116, 144]],
    [0.64, [16, 185, 129]],
    [0.82, [245, 158, 11]],
    [1.0, [254, 226, 226]],
  ];

  for (let i = 0; i < 256; i += 1) {
    const t = i / 255;
    let s0 = stops[0];
    let s1 = stops[stops.length - 1];
    for (let k = 0; k < stops.length - 1; k += 1) {
      if (t >= stops[k][0] && t <= stops[k + 1][0]) {
        s0 = stops[k];
        s1 = stops[k + 1];
        break;
      }
    }
    const span = Math.max(1e-6, s1[0] - s0[0]);
    const f = (t - s0[0]) / span;
    lut[i * 3] = Math.round(s0[1][0] + f * (s1[1][0] - s0[1][0]));
    lut[i * 3 + 1] = Math.round(s0[1][1] + f * (s1[1][1] - s0[1][1]));
    lut[i * 3 + 2] = Math.round(s0[1][2] + f * (s1[1][2] - s0[1][2]));
  }
  return lut;
})();

const FREQ_TICKS = [4000, 2000, 1000, 500, 100, 50];

export function SpectrogramCanvas({
  spectrogram,
  peakFreqHz,
  activeClass,
  confidence,
}: SpectrogramCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const offscreenRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    if (!offscreenRef.current) {
      const off = document.createElement('canvas');
      off.width = FRAMES;
      off.height = MEL_BINS;
      offscreenRef.current = off;
    }

    const offscreen = offscreenRef.current;
    const offCtx = offscreen.getContext('2d');
    if (!offCtx) return;

    const width = canvas.width;
    const height = canvas.height;

    ctx.fillStyle = '#070b11';
    ctx.fillRect(0, 0, width, height);

    if (spectrogram && spectrogram.length === MEL_BINS * FRAMES) {
      const imgData = offCtx.createImageData(FRAMES, MEL_BINS);
      const pixels = imgData.data;

      // Map z-scored log-mel values [-2.2, +3.2] into [0..255]
      const minZ = -2.2;
      const invRange = 255 / (3.2 - minZ);

      for (let m = 0; m < MEL_BINS; m += 1) {
        // Flip Y so 50 Hz (bin 0) sits at bottom and 4000 Hz (bin 127) sits at top
        const canvasY = MEL_BINS - 1 - m;
        const rowSrc = m * FRAMES;
        const rowDst = canvasY * FRAMES;

        for (let t = 0; t < FRAMES; t += 1) {
          const val = spectrogram[rowSrc + t];
          const lutIdx = Math.max(0, Math.min(255, Math.round((val - minZ) * invRange))) * 3;
          const pIdx = (rowDst + t) * 4;
          pixels[pIdx] = COLOR_LUT[lutIdx];
          pixels[pIdx + 1] = COLOR_LUT[lutIdx + 1];
          pixels[pIdx + 2] = COLOR_LUT[lutIdx + 2];
          pixels[pIdx + 3] = 255;
        }
      }

      offCtx.putImageData(imgData, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(offscreen, 0, 0, width, height);
    } else {
      ctx.fillStyle = 'rgba(148, 163, 184, 0.35)';
      ctx.font = '600 13px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.textAlign = 'center';
      ctx.fillText('AWAITING 16 kHz ACOUSTIC STREAM', width / 2, height / 2);
    }

    // Frequency gridlines (HTK Mel vertical mapping)
    const melMin = hzToMel(DSP_CONFIG.fMin);
    const melMax = hzToMel(DSP_CONFIG.fMax);
    const melSpan = melMax - melMin;

    ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'left';

    for (const freq of FREQ_TICKS) {
      const norm = (hzToMel(freq) - melMin) / melSpan;
      const y = Math.round(height * (1 - norm));
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.22)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, y + 0.5);
      ctx.lineTo(width, y + 0.5);
      ctx.stroke();

      ctx.fillStyle = 'rgba(226, 232, 240, 0.72)';
      const labelY = y < 14 ? y + 11 : y - 3;
      ctx.fillText(`${freq} Hz`, 8, labelY);
    }

    // Peak frequency crosshair
    if (spectrogram && peakFreqHz >= DSP_CONFIG.fMin && peakFreqHz <= DSP_CONFIG.fMax) {
      const peakNorm = (hzToMel(peakFreqHz) - melMin) / melSpan;
      const peakY = Math.round(height * (1 - peakNorm));
      ctx.strokeStyle = 'rgba(34, 211, 238, 0.9)';
      ctx.setLineDash([5, 4]);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(0, peakY + 0.5);
      ctx.lineTo(width, peakY + 0.5);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = 'rgba(8, 15, 24, 0.85)';
      ctx.fillRect(width - 102, Math.max(4, peakY - 16), 94, 16);
      ctx.fillStyle = '#22d3ee';
      ctx.textAlign = 'right';
      ctx.fillText(`PEAK ${peakFreqHz.toFixed(1)} Hz`, width - 12, Math.max(15, peakY - 4));
    }
  }, [spectrogram, peakFreqHz]);

  return (
    <div className="spectrogram-frame">
      <div className="spectrogram-header">
        <span className="panel-title">LOG-MEL SPECTROGRAM (50–4000 Hz · 128×100)</span>
        <span className="spectrogram-badge">
          {activeClass
            ? `${activeClass} (${(confidence * 100).toFixed(0)}%)`
            : 'DSP Standby'}
        </span>
      </div>
      <canvas
        ref={canvasRef}
        width={640}
        height={240}
        className="spectrogram-canvas"
        aria-label="Live 128-bin Log-Mel Spectrogram"
      />
      <div className="spectrogram-axis-footer">
        <span>t = -1.00 s</span>
        <span>-0.50 s</span>
        <span>0.00 s (LIVE)</span>
      </div>
    </div>
  );
}
