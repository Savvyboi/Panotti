const EARTH_RADIUS_M = 6_371_000;
const DEG_TO_RAD = Math.PI / 180;
const RAD_TO_DEG = 180 / Math.PI;

export interface LocalPointMeters {
  x: number;
  y: number;
}

export interface GeoPoint {
  lat: number;
  lng: number;
}

export interface KalmanState {
  lat: number;
  lng: number;
  vx: number;
  vy: number;
  speedMps: number;
  headingDeg: number;
}

export interface SerializedKalmanFilter2D {
  originLat: number;
  originLng: number;
  state: [number, number, number, number];
  covariance: number[];
  processNoiseAccelMps2: number;
}

export interface KalmanFilterOptions {
  initialVx?: number;
  initialVy?: number;
  initialPosVarM2?: number;
  initialVelVarM2S2?: number;
  processNoiseAccelMps2?: number;
}

/**
 * Projects WGS84 (lat, lng) into local tangent-plane meters (x = East, y = North)
 * centered at (originLat, originLng).
 */
export function latLngToLocalMeters(
  lat: number,
  lng: number,
  originLat: number,
  originLng: number
): LocalPointMeters {
  const cosOriginLat = Math.max(Math.cos(originLat * DEG_TO_RAD), 1e-6);
  const x = (lng - originLng) * DEG_TO_RAD * EARTH_RADIUS_M * cosOriginLat;
  const y = (lat - originLat) * DEG_TO_RAD * EARTH_RADIUS_M;
  return { x, y };
}

/**
 * Projects local tangent-plane meters (x = East, y = North) back to WGS84 (lat, lng)
 * centered at (originLat, originLng).
 */
export function localMetersToLatLng(
  x: number,
  y: number,
  originLat: number,
  originLng: number
): GeoPoint {
  const cosOriginLat = Math.max(Math.cos(originLat * DEG_TO_RAD), 1e-6);
  const lat = originLat + (y / EARTH_RADIUS_M) * RAD_TO_DEG;
  const lng = originLng + (x / (EARTH_RADIUS_M * cosOriginLat)) * RAD_TO_DEG;
  return { lat, lng };
}

/**
 * Constant-velocity 2D Kalman filter with state vector [x, y, vx, vy]^T
 * in local tangent-plane meters and Joseph-form covariance update.
 */
export class KalmanFilter2D {
  readonly originLat: number;
  readonly originLng: number;
  private x: [number, number, number, number];
  private P: number[]; // Row-major 4x4 covariance matrix (length 16)
  private readonly processNoiseAccelMps2: number;

  constructor(
    originLat: number,
    originLng: number,
    initialX = 0,
    initialY = 0,
    options: KalmanFilterOptions = {}
  ) {
    this.originLat = originLat;
    this.originLng = originLng;
    const vx = options.initialVx ?? 0;
    const vy = options.initialVy ?? 0;
    const posVar = options.initialPosVarM2 ?? 900; // 30m initial position std dev
    const velVar = options.initialVelVarM2S2 ?? 3_600; // 60 m/s std dev
    this.processNoiseAccelMps2 = options.processNoiseAccelMps2 ?? 3.0;

    this.x = [initialX, initialY, vx, vy];
    this.P = [
      posVar, 0, 0, 0,
      0, posVar, 0, 0,
      0, 0, velVar, 0,
      0, 0, 0, velVar
    ];
  }

  /**
   * Advances state and covariance forward by dtSec seconds using constant-velocity kinematics
   * and continuous white-noise acceleration process covariance Q(dt).
   */
  predict(dtSec: number): void {
    if (!Number.isFinite(dtSec) || dtSec <= 0) {
      return;
    }

    const dt = dtSec;
    const [px, py, vx, vy] = this.x;
    this.x = [px + vx * dt, py + vy * dt, vx, vy];

    // F * P * F^T computed explicitly for F = [[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]]
    const P = this.P;
    const F = [
      1, 0, dt, 0,
      0, 1, 0, dt,
      0, 0, 1, 0,
      0, 0, 0, 1
    ];

    const FP = matMul4x4(F, P);
    const FT = transpose4x4(F);
    const nextP = matMul4x4(FP, FT);

    // Continuous white noise acceleration model Q
    const q = this.processNoiseAccelMps2 * this.processNoiseAccelMps2;
    const dt2 = dt * dt;
    const dt3 = dt2 * dt;
    const dt4 = dt2 * dt2;
    const qPosPos = 0.25 * dt4 * q;
    const qPosVel = 0.5 * dt3 * q;
    const qVelVel = dt2 * q;

    nextP[0] += qPosPos;
    nextP[2] += qPosVel;
    nextP[8] += qPosVel;
    nextP[10] += qVelVel;

    nextP[5] += qPosPos;
    nextP[7] += qPosVel;
    nextP[13] += qPosVel;
    nextP[15] += qVelVel;

    symmetrize4x4InPlace(nextP);
    this.P = nextP;
  }

  /**
   * Incorporates a 2D position measurement (measX, measY) with isotropic variance measVarM2
   * using the numerically stable Joseph-form covariance update:
   * P_new = (I - K H) P (I - K H)^T + K R K^T.
   */
  update(measX: number, measY: number, measVarM2: number): void {
    if (!Number.isFinite(measX) || !Number.isFinite(measY)) {
      return;
    }

    const r = Number.isFinite(measVarM2) ? Math.max(measVarM2, 1e-3) : 10_000;
    const P = this.P;

    // Innovation y = z - Hx where H = [[1,0,0,0],[0,1,0,0]]
    const innovX = measX - this.x[0];
    const innovY = measY - this.x[1];

    // Innovation covariance S = H P H^T + R (2x2)
    const s00 = P[0] + r;
    const s01 = P[1];
    const s10 = P[4];
    const s11 = P[5] + r;
    const det = s00 * s11 - s01 * s10;
    if (det <= 1e-12) {
      return;
    }

    const invDet = 1 / det;
    const invS00 = s11 * invDet;
    const invS01 = -s01 * invDet;
    const invS10 = -s10 * invDet;
    const invS11 = s00 * invDet;

    // Kalman gain K = P H^T S^-1 (4x2 row-major)
    // P H^T has rows [P[i*4 + 0], P[i*4 + 1]]
    const K = new Array<number>(8);
    for (let i = 0; i < 4; i++) {
      const p0 = P[i * 4];
      const p1 = P[i * 4 + 1];
      K[i * 2] = p0 * invS00 + p1 * invS10;
      K[i * 2 + 1] = p0 * invS01 + p1 * invS11;
    }

    // State update: x = x + K * innov
    for (let i = 0; i < 4; i++) {
      this.x[i] += K[i * 2] * innovX + K[i * 2 + 1] * innovY;
    }

    // Joseph form: A = I - K H (4x4)
    // Since H only selects columns 0 and 1, (K H) has columns 0 and 1 equal to K, cols 2 and 3 zero.
    const A = [
      1 - K[0], -K[1], 0, 0,
      -K[2], 1 - K[3], 0, 0,
      -K[4], -K[5], 1, 0,
      -K[6], -K[7], 0, 1
    ];

    const AP = matMul4x4(A, P);
    const AT = transpose4x4(A);
    const APAT = matMul4x4(AP, AT);

    // K * R * K^T where R = diag(r, r)
    const nextP = new Array<number>(16);
    for (let i = 0; i < 4; i++) {
      const ki0 = K[i * 2];
      const ki1 = K[i * 2 + 1];
      for (let j = 0; j < 4; j++) {
        const kj0 = K[j * 2];
        const kj1 = K[j * 2 + 1];
        nextP[i * 4 + j] = APAT[i * 4 + j] + r * (ki0 * kj0 + ki1 * kj1);
      }
    }

    symmetrize4x4InPlace(nextP);
    this.P = nextP;
  }

  getLocalPosition(): LocalPointMeters {
    return { x: this.x[0], y: this.x[1] };
  }

  getCovariance(): number[] {
    return [...this.P];
  }

  getState(): KalmanState {
    const { lat, lng } = localMetersToLatLng(
      this.x[0],
      this.x[1],
      this.originLat,
      this.originLng
    );
    const vx = this.x[2];
    const vy = this.x[3];
    const speedMps = Math.hypot(vx, vy);
    const headingDeg =
      speedMps > 1e-6 ? ((Math.atan2(vx, vy) * RAD_TO_DEG) + 360) % 360 : 0;

    return { lat, lng, vx, vy, speedMps, headingDeg };
  }

  serialize(): SerializedKalmanFilter2D {
    return {
      originLat: this.originLat,
      originLng: this.originLng,
      state: [...this.x] as [number, number, number, number],
      covariance: [...this.P],
      processNoiseAccelMps2: this.processNoiseAccelMps2
    };
  }

  static deserialize(data: SerializedKalmanFilter2D): KalmanFilter2D {
    const kf = new KalmanFilter2D(
      data.originLat,
      data.originLng,
      data.state[0],
      data.state[1],
      {
        initialVx: data.state[2],
        initialVy: data.state[3],
        processNoiseAccelMps2: data.processNoiseAccelMps2
      }
    );
    if (Array.isArray(data.covariance) && data.covariance.length === 16) {
      kf.P = [...data.covariance];
    }
    return kf;
  }
}

function matMul4x4(a: readonly number[], b: readonly number[]): number[] {
  const out = new Array<number>(16);
  for (let i = 0; i < 4; i++) {
    const row = i * 4;
    for (let j = 0; j < 4; j++) {
      out[row + j] =
        a[row] * b[j] +
        a[row + 1] * b[4 + j] +
        a[row + 2] * b[8 + j] +
        a[row + 3] * b[12 + j];
    }
  }
  return out;
}

function transpose4x4(m: readonly number[]): number[] {
  return [
    m[0], m[4], m[8], m[12],
    m[1], m[5], m[9], m[13],
    m[2], m[6], m[10], m[14],
    m[3], m[7], m[11], m[15]
  ];
}

function symmetrize4x4InPlace(m: number[]): void {
  for (let i = 0; i < 4; i++) {
    for (let j = i + 1; j < 4; j++) {
      const avg = 0.5 * (m[i * 4 + j] + m[j * 4 + i]);
      m[i * 4 + j] = avg;
      m[j * 4 + i] = avg;
    }
  }
}
