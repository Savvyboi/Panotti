import type { SignalMetrics } from '../types';

export const DSP_CONFIG = {
  sampleRate: 16000,
  nFft: 512,
  hopLength: 160,
  numFrames: 100,
  nMels: 128,
  fMin: 50.0,
  fMax: 4000.0,
  paddedLength: 16000 + (512 - 160), // 16352
  numBins: 512 / 2 + 1, // 257
  freqStepHz: 16000 / 512, // 31.25 Hz
  eps: 1e-6,
} as const;

export function hzToMel(hz: number): number {
  return 2595 * Math.log10(1 + hz / 700);
}

export function melToHz(mel: number): number {
  return 700 * (Math.pow(10, mel / 2595) - 1);
}

/**
 * Builds the 512-point periodic Hann window:
 * w[n] = 0.5 * (1 - cos(2 * pi * n / 512)) for n = 0..511.
 */
export function buildPeriodicHannWindow(nFft = DSP_CONFIG.nFft): Float64Array {
  const win = new Float64Array(nFft);
  for (let n = 0; n < nFft; n += 1) {
    win[n] = 0.5 * (1 - Math.cos((2 * Math.PI * n) / nFft));
  }
  return win;
}

/**
 * Builds the 128x257 HTK Mel filterbank matrix with narrow-band center-interpolation
 * fallback and L1 row normalization, matching pipelines/dataset_builder/feature_extraction.py.
 */
export function buildMelFilterbank(
  nMels = DSP_CONFIG.nMels,
  nFft = DSP_CONFIG.nFft,
  sampleRate = DSP_CONFIG.sampleRate,
  fMin = DSP_CONFIG.fMin,
  fMax = DSP_CONFIG.fMax
): Float64Array {
  const nBins = nFft / 2 + 1;
  const df = sampleRate / nFft;
  const filterbank = new Float64Array(nMels * nBins);

  const melMin = hzToMel(fMin);
  const melMax = hzToMel(fMax);
  const numPoints = nMels + 2; // 130
  const hzPoints = new Float64Array(numPoints);

  for (let i = 0; i < numPoints; i += 1) {
    const mel = melMin + (i * (melMax - melMin)) / (numPoints - 1);
    hzPoints[i] = melToHz(mel);
  }

  for (let m = 0; m < nMels; m += 1) {
    const fLeft = hzPoints[m];
    const fCenter = hzPoints[m + 1];
    const fRight = hzPoints[m + 2];
    const rowOffset = m * nBins;

    let rowSum = 0;
    for (let k = 0; k < nBins; k += 1) {
      const f = k * df;
      let weight = 0;
      if (f >= fLeft && f <= fCenter && fCenter > fLeft) {
        weight = (f - fLeft) / (fCenter - fLeft);
      } else if (f > fCenter && f <= fRight && fRight > fCenter) {
        weight = (fRight - f) / (fRight - fCenter);
      }
      filterbank[rowOffset + k] = weight;
      rowSum += weight;
    }

    // Narrow-band fix when triangular support falls strictly between adjacent FFT bins
    if (rowSum === 0) {
      const exactBin = fCenter / df;
      const k0 = Math.max(0, Math.min(nBins - 1, Math.floor(exactBin)));
      const k1 = Math.min(k0 + 1, nBins - 1);
      const frac = Math.max(0, Math.min(1, exactBin - k0));

      if (k1 === k0) {
        filterbank[rowOffset + k0] = 1;
      } else {
        filterbank[rowOffset + k0] = 1 - frac;
        filterbank[rowOffset + k1] = frac;
      }
      rowSum = 1;
    }

    if (rowSum > 0) {
      const invSum = 1 / rowSum;
      for (let k = 0; k < nBins; k += 1) {
        filterbank[rowOffset + k] *= invSum;
      }
    }
  }

  return filterbank;
}

class Radix2Fft512 {
  public readonly n = 512;
  private readonly bitRev = new Uint16Array(512);
  private readonly cosTable = new Float64Array(256);
  private readonly sinTable = new Float64Array(256);

  constructor() {
    const bits = 9; // log2(512)
    for (let i = 0; i < this.n; i += 1) {
      let rev = 0;
      let x = i;
      for (let b = 0; b < bits; b += 1) {
        rev = (rev << 1) | (x & 1);
        x >>= 1;
      }
      this.bitRev[i] = rev;
    }

    for (let k = 0; k < this.n / 2; k += 1) {
      const angle = (-2 * Math.PI * k) / this.n;
      this.cosTable[k] = Math.cos(angle);
      this.sinTable[k] = Math.sin(angle);
    }
  }

  public powerSpectrum(realInput: Float64Array, outPower: Float64Array, imScratch: Float64Array): void {
    const n = this.n;

    // Bit-reversed copy into imScratch=0 and realInput reordered via temporary swap
    for (let i = 0; i < n; i += 1) {
      const j = this.bitRev[i];
      if (i < j) {
        const tmp = realInput[i];
        realInput[i] = realInput[j];
        realInput[j] = tmp;
      }
      imScratch[i] = 0;
    }

    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1;
      const stride = n / size;
      for (let i = 0; i < n; i += size) {
        for (let j = 0; j < half; j += 1) {
          const twIdx = j * stride;
          const wr = this.cosTable[twIdx];
          const wi = this.sinTable[twIdx];

          const evenIdx = i + j;
          const oddIdx = evenIdx + half;

          const oddRe = realInput[oddIdx];
          const oddIm = imScratch[oddIdx];

          const tRe = oddRe * wr - oddIm * wi;
          const tIm = oddRe * wi + oddIm * wr;

          const evenRe = realInput[evenIdx];
          const evenIm = imScratch[evenIdx];

          realInput[oddIdx] = evenRe - tRe;
          imScratch[oddIdx] = evenIm - tIm;
          realInput[evenIdx] = evenRe + tRe;
          imScratch[evenIdx] = evenIm + tIm;
        }
      }
    }

    const nBins = n / 2 + 1; // 257
    for (let k = 0; k < nBins; k += 1) {
      const re = realInput[k];
      const im = imScratch[k];
      outPower[k] = re * re + im * im;
    }
  }
}

/**
 * Computes exact 128x100 z-scored Log-Mel spectrogram matching
 * `pipelines/dataset_builder/feature_extraction.py`.
 */
export class MelSpectrogramExtractor {
  private readonly window: Float64Array;
  private readonly melFilterbank: Float64Array;
  private readonly fft: Radix2Fft512;

  // Reusable scratch buffers to minimize GC pressure during 4 Hz inference
  private readonly paddedSignal = new Float64Array(DSP_CONFIG.paddedLength);
  private readonly frameRe = new Float64Array(DSP_CONFIG.nFft);
  private readonly frameIm = new Float64Array(DSP_CONFIG.nFft);
  private readonly framePower = new Float64Array(DSP_CONFIG.numBins);
  private readonly logMelScratch = new Float64Array(DSP_CONFIG.nMels * DSP_CONFIG.numFrames);
  private readonly avgPowerScratch = new Float64Array(DSP_CONFIG.numBins);

  constructor() {
    this.window = buildPeriodicHannWindow(DSP_CONFIG.nFft);
    this.melFilterbank = buildMelFilterbank();
    this.fft = new Radix2Fft512();
  }

  public getMelFilterbank(): Float64Array {
    return this.melFilterbank;
  }

  private preparePaddedSignal(pcm16k: Float32Array): void {
    this.paddedSignal.fill(0);
    const copyLen = Math.min(pcm16k.length, DSP_CONFIG.sampleRate);
    for (let i = 0; i < copyLen; i += 1) {
      this.paddedSignal[i] = pcm16k[i];
    }
  }

  /**
   * Extract normalized [1, 1, 128, 100] Log-Mel spectrogram as row-major Float32Array(12800).
   */
  public extract(pcm16k: Float32Array): Float32Array {
    const { nFft, hopLength, numFrames, nMels, numBins, eps } = DSP_CONFIG;
    this.preparePaddedSignal(pcm16k);

    let sum = 0;
    for (let t = 0; t < numFrames; t += 1) {
      const offset = t * hopLength;
      for (let n = 0; n < nFft; n += 1) {
        this.frameRe[n] = this.paddedSignal[offset + n] * this.window[n];
      }

      this.fft.powerSpectrum(this.frameRe, this.framePower, this.frameIm);

      for (let m = 0; m < nMels; m += 1) {
        const fbOffset = m * numBins;
        let melEnergy = 0;
        for (let k = 0; k < numBins; k += 1) {
          melEnergy += this.melFilterbank[fbOffset + k] * this.framePower[k];
        }
        const logVal = Math.log(Math.max(melEnergy, eps));
        this.logMelScratch[m * numFrames + t] = logVal;
        sum += logVal;
      }
    }

    const totalElements = nMels * numFrames;
    const mean = sum / totalElements;

    let sqDiffSum = 0;
    for (let i = 0; i < totalElements; i += 1) {
      const diff = this.logMelScratch[i] - mean;
      sqDiffSum += diff * diff;
    }
    const std = Math.sqrt(sqDiffSum / totalElements);
    const invDenom = 1 / (std + eps);

    const output = new Float32Array(totalElements);
    for (let i = 0; i < totalElements; i += 1) {
      output[i] = (this.logMelScratch[i] - mean) * invDenom;
    }

    return output;
  }

  /**
   * Estimates dominant frequency (Hz via parabolic bin interpolation in 50..4000 Hz)
   * and broadband SNR (dB against median spectral noise floor).
   */
  public estimateSignalMetrics(pcm16k: Float32Array): SignalMetrics {
    const { nFft, hopLength, numFrames, numBins, freqStepHz } = DSP_CONFIG;
    this.preparePaddedSignal(pcm16k);
    this.avgPowerScratch.fill(0);

    for (let t = 0; t < numFrames; t += 1) {
      const offset = t * hopLength;
      for (let n = 0; n < nFft; n += 1) {
        this.frameRe[n] = this.paddedSignal[offset + n] * this.window[n];
      }
      this.fft.powerSpectrum(this.frameRe, this.framePower, this.frameIm);
      for (let k = 0; k < numBins; k += 1) {
        this.avgPowerScratch[k] += this.framePower[k];
      }
    }

    const minBin = 2; // 62.5 Hz (closest bin above 50 Hz; ignores DC and 31.25 Hz sub-band)
    const maxBin = 128; // 4000 Hz

    let peakBin = minBin;
    let peakPower = 0;
    let totalBandPower = 0;
    const bandValues: number[] = [];

    for (let k = minBin; k <= maxBin; k += 1) {
      const avgP = this.avgPowerScratch[k] / numFrames;
      totalBandPower += avgP;
      bandValues.push(avgP);
      if (avgP > peakPower) {
        peakPower = avgP;
        peakBin = k;
      }
    }

    if (peakPower <= 1e-12) {
      return { snr_db: -30, peak_freq_hz: 0 };
    }

    // Parabolic interpolation on log power for sub-bin frequency resolution
    let delta = 0;
    if (peakBin > minBin && peakBin < maxBin) {
      const alpha = Math.log(Math.max(this.avgPowerScratch[peakBin - 1] / numFrames, 1e-18));
      const beta = Math.log(Math.max(peakPower, 1e-18));
      const gamma = Math.log(Math.max(this.avgPowerScratch[peakBin + 1] / numFrames, 1e-18));
      const denom = alpha - 2 * beta + gamma;
      if (Math.abs(denom) > 1e-12) {
        delta = Math.max(-0.5, Math.min(0.5, (0.5 * (alpha - gamma)) / denom));
      }
    }

    const peakFreqHz = Number(((peakBin + delta) * freqStepHz).toFixed(1));

    bandValues.sort((a, b) => a - b);
    const medianNoiseBin = bandValues[Math.floor(bandValues.length / 2)] || 1e-12;
    const noiseFloorTotal = medianNoiseBin * bandValues.length;
    const excessSignalPower = Math.max(totalBandPower - noiseFloorTotal, 1e-12);
    const rawSnrDb = 10 * Math.log10(excessSignalPower / Math.max(noiseFloorTotal, 1e-12));
    const snrDb = Number(Math.max(-30, Math.min(60, rawSnrDb)).toFixed(1));

    return {
      snr_db: snrDb,
      peak_freq_hz: peakFreqHz,
    };
  }
}

const defaultExtractor = new MelSpectrogramExtractor();

export function extractLogMelSpectrogram(pcm16k: Float32Array): Float32Array {
  return defaultExtractor.extract(pcm16k);
}

export function estimateSignalMetrics(pcm16k: Float32Array): SignalMetrics {
  return defaultExtractor.estimateSignalMetrics(pcm16k);
}
