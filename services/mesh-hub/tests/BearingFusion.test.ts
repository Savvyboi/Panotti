import { describe, expect, it } from "vitest";
import {
  BearingFusion,
  computeMultiNodeBearing,
  snrToWeight
} from "../src/tracker/BearingFusion";
import {
  latLngToLocalMeters,
  localMetersToLatLng
} from "../src/tracker/KalmanFilter2D";
import type { DetectionClass, NormalizedDetectionMessage } from "../src/types";

const BASE_LAT = 50.4501;
const BASE_LNG = 30.5234;

function offsetPoint(eastMeters: number, northMeters: number) {
  return localMetersToLatLng(eastMeters, northMeters, BASE_LAT, BASE_LNG);
}

function makeDetection(params: {
  nodeId: string;
  timestamp: number;
  eastM?: number;
  northM?: number;
  uavClass?: DetectionClass;
  confidence?: number;
  snrDb?: number;
  peakFreqHz?: number;
  accuracy?: number;
}): NormalizedDetectionMessage {
  const { lat, lng } = offsetPoint(params.eastM ?? 0, params.northM ?? 0);
  return {
    type: "detection",
    nodeId: params.nodeId,
    timestamp: params.timestamp,
    location: {
      lat,
      lng,
      accuracy: params.accuracy ?? 15
    },
    detection: {
      class: params.uavClass ?? "Shahed_Piston_UAV",
      confidence: params.confidence ?? 0.9,
      snr_db: params.snrDb ?? 15,
      peak_freq_hz: params.peakFreqHz ?? 92.5
    }
  };
}

describe("BearingFusion", () => {
  it("clamps SNR weights to [-30, 60] dB range", () => {
    expect(snrToWeight(-100)).toBeCloseTo(1e-3, 12);
    expect(snrToWeight(-30)).toBeCloseTo(1e-3, 12);
    expect(snrToWeight(20)).toBeCloseTo(100, 12);
    expect(snrToWeight(60)).toBeCloseTo(1e6, 6);
    expect(snrToWeight(150)).toBeCloseTo(1e6, 6);
  });

  it("clusters same-class UAV detections within 3000 ms and 8000 m with SNR-weighted centroid", () => {
    const fusion = new BearingFusion({ originLat: BASE_LAT, originLng: BASE_LNG });

    // Node 1 at (0, 0) with SNR 10 dB -> weight = 10
    const res1 = fusion.ingest(
      makeDetection({
        nodeId: "node-1",
        timestamp: 10_000,
        eastM: 0,
        northM: 0,
        snrDb: 10
      })
    );
    expect(res1?.track.nodeCount).toBe(1);

    // Node 2 at (2200 m, 0) with SNR 20 dB -> weight = 100
    const res2 = fusion.ingest(
      makeDetection({
        nodeId: "node-2",
        timestamp: 10_500,
        eastM: 2200,
        northM: 0,
        snrDb: 20
      })
    );

    expect(res2).not.toBeNull();
    expect(fusion.getTracks()).toHaveLength(1);
    expect(res2!.cluster.nodeCount).toBe(2);

    // Expected centroid x = (10 * 0 + 100 * 2200) / 110 = 2000 m
    const expectedX = (10 * 0 + 100 * 2200) / 110;
    expect(res2!.cluster.centroidX).toBeCloseTo(expectedX, 3);
    expect(res2!.cluster.centroidY).toBeCloseTo(0, 3);
    expect(res2!.cluster.bearingDeg).toBeCloseTo(90, 1); // Eastward
  });

  it("computes multi-node arrival-time & energy bearing estimate for >= 2 distinct nodes", () => {
    const single = computeMultiNodeBearing(
      [{ x: 0, y: 0, timestamp: 1000, weight: 10 }],
      0,
      0
    );
    expect(single).toBeNull();

    // Node A at South (0, 0) hears late (t=1600) and weak (w=5)
    // Node B at North (0, 1000) hears early (t=1000) and strong (w=95)
    // Source direction from array is due North (0 deg)
    const bearingNorth = computeMultiNodeBearing(
      [
        { x: 0, y: 0, timestamp: 1600, weight: 5 },
        { x: 0, y: 1000, timestamp: 1000, weight: 95 }
      ],
      0,
      950
    );
    expect(bearingNorth).not.toBeNull();
    expect(bearingNorth!).toBeCloseTo(0, 1);

    // TDOA-only (equal weights): Node B at West (-1000, 0) hears earlier than East (1000, 0)
    const bearingWest = computeMultiNodeBearing(
      [
        { x: 1000, y: 0, timestamp: 2000, weight: 10 },
        { x: -1000, y: 0, timestamp: 1000, weight: 10 }
      ],
      0,
      0
    );
    expect(bearingWest).toBeCloseTo(270, 1);
  });

  it("separates different UAV classes and ignores non-UAV classes", () => {
    const fusion = new BearingFusion({ originLat: BASE_LAT, originLng: BASE_LNG });

    const bg = fusion.ingest(
      makeDetection({
        nodeId: "node-bg",
        timestamp: 1000,
        uavClass: "Environmental_Background"
      })
    );
    const ground = fusion.ingest(
      makeDetection({
        nodeId: "node-gnd",
        timestamp: 1100,
        uavClass: "Ground_Combustion_Noise"
      })
    );
    expect(bg).toBeNull();
    expect(ground).toBeNull();
    expect(fusion.getTracks()).toHaveLength(0);

    fusion.ingest(
      makeDetection({
        nodeId: "node-1",
        timestamp: 1200,
        eastM: 0,
        northM: 0,
        uavClass: "Shahed_Piston_UAV"
      })
    );
    fusion.ingest(
      makeDetection({
        nodeId: "node-2",
        timestamp: 1300,
        eastM: 50,
        northM: 50,
        uavClass: "Multirotor_UAV"
      })
    );

    expect(fusion.getTracks()).toHaveLength(2);
  });

  it("does not cluster detections beyond 8000 m or 3000 ms and enforces 1500 m association gate", () => {
    const fusion = new BearingFusion({ originLat: BASE_LAT, originLng: BASE_LNG });

    fusion.ingest(
      makeDetection({
        nodeId: "node-1",
        timestamp: 10_000,
        eastM: 0,
        northM: 0
      })
    );

    // Within 3000 ms, but 15000 m away (> 8000 m cluster radius and > 1500 m gate)
    fusion.ingest(
      makeDetection({
        nodeId: "node-far",
        timestamp: 11_000,
        eastM: 15_000,
        northM: 0
      })
    );
    expect(fusion.getTracks()).toHaveLength(2);

    // Outside 3000 ms window (14_001 - 10_000 = 4001 ms) and 2000 m from first track (> 1500 m gate)
    fusion.ingest(
      makeDetection({
        nodeId: "node-mid",
        timestamp: 14_001,
        eastM: 2000,
        northM: 0
      })
    );
    expect(fusion.getTracks()).toHaveLength(3);
  });

  it("caps track history at 60 points and prunes tracks stale beyond 15000 ms", () => {
    const fusion = new BearingFusion({ originLat: BASE_LAT, originLng: BASE_LNG });

    for (let i = 0; i < 75; i++) {
      fusion.ingest(
        makeDetection({
          nodeId: "node-1",
          timestamp: 1000 + i * 500,
          eastM: i * 10,
          northM: 0
        })
      );
    }

    const tracks = fusion.getTracks();
    expect(tracks).toHaveLength(1);
    expect(tracks[0].history).toHaveLength(60);

    // Latest point in history corresponds to i = 74 (timestamp = 38000)
    expect(tracks[0].history[59].timestamp).toBe(1000 + 74 * 500);
    expect(tracks[0].history[0].timestamp).toBe(1000 + 15 * 500);

    // Check smooth local coordinates near end of trajectory
    const lastLocal = latLngToLocalMeters(
      tracks[0].lat,
      tracks[0].lng,
      BASE_LAT,
      BASE_LNG
    );
    expect(lastLocal.x).toBeGreaterThan(700);

    // Prune before 15000 ms -> track remains
    expect(fusion.prune(tracks[0].lastUpdated + 15_000)).toBe(false);
    expect(fusion.getTracks()).toHaveLength(1);

    // Prune beyond 15000 ms -> track removed
    expect(fusion.prune(tracks[0].lastUpdated + 15_001)).toBe(true);
    expect(fusion.getTracks()).toHaveLength(0);
  });

  it("serializes and restores fusion buffer and tracks cleanly", () => {
    const fusion = new BearingFusion({ originLat: BASE_LAT, originLng: BASE_LNG });
    fusion.ingest(
      makeDetection({ nodeId: "node-1", timestamp: 5000, eastM: 100, northM: 200 })
    );
    fusion.ingest(
      makeDetection({ nodeId: "node-2", timestamp: 5400, eastM: 300, northM: 200 })
    );

    const snapshot = JSON.parse(JSON.stringify(fusion.serialize()));
    const restored = BearingFusion.restore(snapshot);

    expect(restored.getTracks()).toEqual(fusion.getTracks());
    expect(restored.getBufferedDetectionCount()).toBe(
      fusion.getBufferedDetectionCount()
    );

    // Subsequent update continues same track
    const next = restored.ingest(
      makeDetection({ nodeId: "node-1", timestamp: 6000, eastM: 150, northM: 220 })
    );
    expect(next?.track.trackId).toBe(fusion.getTracks()[0].trackId);
  });
});
