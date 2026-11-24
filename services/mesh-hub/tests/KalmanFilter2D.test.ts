import { describe, expect, it } from "vitest";
import {
  KalmanFilter2D,
  latLngToLocalMeters,
  localMetersToLatLng
} from "../src/tracker/KalmanFilter2D";

describe("KalmanFilter2D", () => {
  const originLat = 50.4501;
  const originLng = 30.5234;

  it("round-trips latLngToLocalMeters and localMetersToLatLng accurately", () => {
    const targetLat = 50.4689;
    const targetLng = 30.5512;

    const local = latLngToLocalMeters(targetLat, targetLng, originLat, originLng);
    expect(local.x).toBeGreaterThan(0); // East of origin
    expect(local.y).toBeGreaterThan(0); // North of origin

    const geo = localMetersToLatLng(local.x, local.y, originLat, originLng);
    expect(geo.lat).toBeCloseTo(targetLat, 9);
    expect(geo.lng).toBeCloseTo(targetLng, 9);
  });

  it("tracks constant-velocity motion and computes speed and compass heading", () => {
    // True trajectory: vx = 40 m/s East, vy = 30 m/s North -> speed = 50 m/s
    // Compass heading (0=N, 90=E) = atan2(40, 30) * 180 / pi ≈ 53.13 deg
    const kf = new KalmanFilter2D(originLat, originLng, 0, 0, {
      initialPosVarM2: 100
    });

    const trueVx = 40;
    const trueVy = 30;
    for (let t = 1; t <= 10; t++) {
      kf.predict(1.0);
      kf.update(trueVx * t, trueVy * t, 100);
    }

    const state = kf.getState();
    expect(state.vx).toBeCloseTo(trueVx, 0);
    expect(state.vy).toBeCloseTo(trueVy, 0);
    expect(state.speedMps).toBeCloseTo(50, 0);

    const expectedHeading = (Math.atan2(trueVx, trueVy) * 180) / Math.PI;
    expect(state.headingDeg).toBeCloseTo(expectedHeading, 0);

    const local = kf.getLocalPosition();
    expect(Math.hypot(local.x - 400, local.y - 300)).toBeLessThan(15);
  });

  it("computes cardinal compass headings correctly", () => {
    const north = new KalmanFilter2D(originLat, originLng, 0, 0, {
      initialVx: 0,
      initialVy: 25
    });
    expect(north.getState().headingDeg).toBeCloseTo(0, 5);

    const east = new KalmanFilter2D(originLat, originLng, 0, 0, {
      initialVx: 25,
      initialVy: 0
    });
    expect(east.getState().headingDeg).toBeCloseTo(90, 5);

    const south = new KalmanFilter2D(originLat, originLng, 0, 0, {
      initialVx: 0,
      initialVy: -25
    });
    expect(south.getState().headingDeg).toBeCloseTo(180, 5);

    const west = new KalmanFilter2D(originLat, originLng, 0, 0, {
      initialVx: -25,
      initialVy: 0
    });
    expect(west.getState().headingDeg).toBeCloseTo(270, 5);
  });

  it("ignores non-positive dtSec in predict", () => {
    const kf = new KalmanFilter2D(originLat, originLng, 100, 200, {
      initialVx: 20,
      initialVy: -10
    });
    const before = kf.getState();
    kf.predict(0);
    kf.predict(-2);
    expect(kf.getState()).toEqual(before);
  });

  it("maintains symmetric positive-definite covariance via Joseph-form update", () => {
    const kf = new KalmanFilter2D(originLat, originLng, 0, 0);
    for (let step = 1; step <= 25; step++) {
      kf.predict(0.5);
      // Small measurement variance stresses standard P=(I-KH)P; Joseph form stays stable.
      kf.update(step * 15, step * 5, 0.05);
    }

    const P = kf.getCovariance();
    for (let i = 0; i < 4; i++) {
      expect(P[i * 4 + i]).toBeGreaterThan(0);
      for (let j = 0; j < 4; j++) {
        expect(P[i * 4 + j]).toBeCloseTo(P[j * 4 + i], 10);
      }
    }
  });

  it("serializes and deserializes without losing filter state or covariance", () => {
    const kf = new KalmanFilter2D(originLat, originLng, 50, -30);
    kf.predict(1.0);
    kf.update(80, -10, 250);
    kf.predict(1.0);
    kf.update(110, 10, 250);

    const serialized = kf.serialize();
    const restored = KalmanFilter2D.deserialize(JSON.parse(JSON.stringify(serialized)));

    expect(restored.getState()).toEqual(kf.getState());
    expect(restored.getCovariance()).toEqual(kf.getCovariance());

    kf.predict(1.0);
    kf.update(140, 30, 250);
    restored.predict(1.0);
    restored.update(140, 30, 250);
    expect(restored.getState()).toEqual(kf.getState());
  });
});
