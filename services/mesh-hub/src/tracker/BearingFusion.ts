import {
  isUavClass,
  type NormalizedDetectionMessage,
  type Track,
  type TrackHistoryPoint,
  type UavClass
} from "../types";
import {
  KalmanFilter2D,
  latLngToLocalMeters,
  localMetersToLatLng,
  type SerializedKalmanFilter2D
} from "./KalmanFilter2D";

export const CLUSTER_WINDOW_MS = 3000;
export const CLUSTER_RADIUS_M = 8000;
export const ASSOCIATION_GATE_M = 1500;
export const MAX_HISTORY_POINTS = 60;
export const TRACK_STALE_MS = 15000;
const SPEED_OF_SOUND_MPS = 343;
const EARTH_RADIUS_M = 6_371_000;
const DEG_TO_RAD = Math.PI / 180;
const RAD_TO_DEG = 180 / Math.PI;

export interface BufferedDetection {
  nodeId: string;
  timestamp: number;
  lat: number;
  lng: number;
  accuracy: number;
  uavClass: UavClass;
  confidence: number;
  snrDb: number;
  peakFreqHz: number;
}

export interface ClusterFusionEstimate {
  uavClass: UavClass;
  centroidLat: number;
  centroidLng: number;
  centroidX: number;
  centroidY: number;
  timestamp: number;
  confidence: number;
  snrDb: number;
  nodeCount: number;
  nodeIds: string[];
  bearingDeg: number | null;
  measVarM2: number;
}

interface InternalTrack {
  trackId: string;
  uavClass: UavClass;
  kalman: KalmanFilter2D;
  confidence: number;
  snrDb: number;
  nodeCount: number;
  contributingNodes: string[];
  lastUpdated: number;
  bearingDeg: number | null;
  history: TrackHistoryPoint[];
}

export interface SerializedTrackRecord {
  trackId: string;
  uavClass: UavClass;
  kalman: SerializedKalmanFilter2D;
  confidence: number;
  snrDb: number;
  nodeCount: number;
  contributingNodes: string[];
  lastUpdated: number;
  bearingDeg: number | null;
  history: TrackHistoryPoint[];
}

export interface SerializedBearingFusion {
  originLat: number | null;
  originLng: number | null;
  latestTimestamp: number;
  nextTrackSeq: number;
  detectionBuffer: BufferedDetection[];
  tracks: SerializedTrackRecord[];
}

export interface BearingFusionOptions {
  originLat?: number;
  originLng?: number;
  clusterWindowMs?: number;
  clusterRadiusM?: number;
  associationGateM?: number;
  maxHistoryPoints?: number;
  trackStaleMs?: number;
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

/**
 * Computes SNR linear energy weight w_i = 10^(clamp(snr_db, -30, 60) / 10).
 */
export function snrToWeight(snrDb: number): number {
  const clampedSnr = clamp(snrDb, -30, 60);
  return Math.pow(10, clampedSnr / 10);
}

/**
 * Great-circle Haversine distance in meters between two WGS84 points.
 */
export function haversineDistanceMeters(
  lat1: number,
  lng1: number,
  lat2: number,
  lng2: number
): number {
  const dLat = (lat2 - lat1) * DEG_TO_RAD;
  const dLng = (lng2 - lng1) * DEG_TO_RAD;
  const radLat1 = lat1 * DEG_TO_RAD;
  const radLat2 = lat2 * DEG_TO_RAD;
  const sinDLat = Math.sin(dLat / 2);
  const sinDLng = Math.sin(dLng / 2);
  const a =
    sinDLat * sinDLat +
    Math.cos(radLat1) * Math.cos(radLat2) * sinDLng * sinDLng;
  return 2 * EARTH_RADIUS_M * Math.atan2(Math.sqrt(a), Math.sqrt(Math.max(0, 1 - a)));
}

/**
 * Computes multi-node arrival-time (TDOA) and energy gradient bearing in degrees
 * (0 = North, 90 = East, 180 = South, 270 = West) when >= 2 distinct nodes contribute.
 */
export function computeMultiNodeBearing(
  nodes: ReadonlyArray<{
    x: number;
    y: number;
    timestamp: number;
    weight: number;
  }>,
  centroidX: number,
  centroidY: number
): number | null {
  if (nodes.length < 2) {
    return null;
  }

  let meanX = 0;
  let meanY = 0;
  for (const n of nodes) {
    meanX += n.x;
    meanY += n.y;
  }
  meanX /= nodes.length;
  meanY /= nodes.length;

  // Energy gradient displacement in meters: points from sensor geometric center toward SNR-weighted centroid.
  const energyVecX = centroidX - meanX;
  const energyVecY = centroidY - meanY;

  // Pairwise acoustic TDOA displacement in meters: earlier-arriving node is closer to source.
  let tdoaWeightedX = 0;
  let tdoaWeightedY = 0;
  let tdoaWeightSum = 0;
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const ni = nodes[i];
      const nj = nodes[j];
      const dx = nj.x - ni.x;
      const dy = nj.y - ni.y;
      const dist = Math.hypot(dx, dy);
      if (dist <= 1) {
        continue;
      }

      const dtSec = (ni.timestamp - nj.timestamp) / 1000;
      if (Math.abs(dtSec) <= 1e-4) {
        continue;
      }

      // If ni arrived later than nj (dtSec > 0), source is shifted toward nj (+dx, +dy).
      const rangeDiff = clamp(SPEED_OF_SOUND_MPS * dtSec, -dist, dist);
      const pairWeight = ni.weight + nj.weight;
      tdoaWeightedX += 0.5 * (dx / dist) * rangeDiff * pairWeight;
      tdoaWeightedY += 0.5 * (dy / dist) * rangeDiff * pairWeight;
      tdoaWeightSum += pairWeight;
    }
  }

  const tdoaVecX = tdoaWeightSum > 0 ? tdoaWeightedX / tdoaWeightSum : 0;
  const tdoaVecY = tdoaWeightSum > 0 ? tdoaWeightedY / tdoaWeightSum : 0;

  const dirX = energyVecX + tdoaVecX;
  const dirY = energyVecY + tdoaVecY;

  if (Math.hypot(dirX, dirY) <= 1e-6) {
    return null;
  }
  return ((Math.atan2(dirX, dirY) * RAD_TO_DEG) + 360) % 360;
}

/**
 * Multi-node acoustic bearing fusion and track association manager.
 * Maintains a rolling 3000 ms detection buffer, clusters same-class UAV detections
 * within 8000 m, computes SNR-weighted centroids and TDOA/energy bearings,
 * and updates 2D Kalman tracks with 1500 m gating and 15000 ms stale pruning.
 */
export class BearingFusion {
  private originLat: number | null;
  private originLng: number | null;
  private readonly clusterWindowMs: number;
  private readonly clusterRadiusM: number;
  private readonly associationGateM: number;
  private readonly maxHistoryPoints: number;
  private readonly trackStaleMs: number;

  private latestTimestamp = 0;
  private nextTrackSeq = 1;
  private detectionBuffer: BufferedDetection[] = [];
  private tracks = new Map<string, InternalTrack>();

  constructor(options: BearingFusionOptions = {}) {
    this.originLat = options.originLat ?? null;
    this.originLng = options.originLng ?? null;
    this.clusterWindowMs = options.clusterWindowMs ?? CLUSTER_WINDOW_MS;
    this.clusterRadiusM = options.clusterRadiusM ?? CLUSTER_RADIUS_M;
    this.associationGateM = options.associationGateM ?? ASSOCIATION_GATE_M;
    this.maxHistoryPoints = options.maxHistoryPoints ?? MAX_HISTORY_POINTS;
    this.trackStaleMs = options.trackStaleMs ?? TRACK_STALE_MS;
  }

  /**
   * Ingests a validated detection message. Non-UAV classes (Ground_Combustion_Noise,
   * Environmental_Background) do not spawn aircraft tracks and return null.
   */
  ingest(message: NormalizedDetectionMessage): {
    cluster: ClusterFusionEstimate;
    track: Track;
  } | null {
    if (!isUavClass(message.detection.class)) {
      return null;
    }

    const uavClass = message.detection.class;
    if (this.originLat === null || this.originLng === null) {
      this.originLat = message.location.lat;
      this.originLng = message.location.lng;
    }

    const now = Math.max(this.latestTimestamp, message.timestamp);
    this.latestTimestamp = now;

    this.upsertDetection({
      nodeId: message.nodeId,
      timestamp: message.timestamp,
      lat: message.location.lat,
      lng: message.location.lng,
      accuracy: message.location.accuracy,
      uavClass,
      confidence: message.detection.confidence,
      snrDb: message.detection.snr_db,
      peakFreqHz: message.detection.peak_freq_hz
    });

    this.pruneExpiredDetections(now);
    this.pruneStaleTracks(now);

    const cluster = this.buildClusterForDetection(message.nodeId, uavClass, message.timestamp);
    if (!cluster) {
      return null;
    }

    const track = this.associateAndUpdateTrack(cluster);
    return { cluster, track };
  }

  /**
   * Prunes detections older than clusterWindowMs and tracks stale beyond trackStaleMs.
   * Returns true if any active tracks were pruned.
   */
  prune(now: number): boolean {
    if (!Number.isFinite(now)) {
      return false;
    }
    this.latestTimestamp = Math.max(this.latestTimestamp, now);
    this.pruneExpiredDetections(now);
    return this.pruneStaleTracks(now);
  }

  getTracks(): Track[] {
    const out: Track[] = [];
    for (const t of this.tracks.values()) {
      out.push(this.toPublicTrack(t));
    }
    return out;
  }

  getBufferedDetectionCount(): number {
    return this.detectionBuffer.length;
  }

  serialize(): SerializedBearingFusion {
    const serializedTracks: SerializedTrackRecord[] = [];
    for (const t of this.tracks.values()) {
      serializedTracks.push({
        trackId: t.trackId,
        uavClass: t.uavClass,
        kalman: t.kalman.serialize(),
        confidence: t.confidence,
        snrDb: t.snrDb,
        nodeCount: t.nodeCount,
        contributingNodes: [...t.contributingNodes],
        lastUpdated: t.lastUpdated,
        bearingDeg: t.bearingDeg,
        history: t.history.map((h) => ({ ...h }))
      });
    }

    return {
      originLat: this.originLat,
      originLng: this.originLng,
      latestTimestamp: this.latestTimestamp,
      nextTrackSeq: this.nextTrackSeq,
      detectionBuffer: this.detectionBuffer.map((d) => ({ ...d })),
      tracks: serializedTracks
    };
  }

  restore(data: SerializedBearingFusion): void {
    this.originLat = data.originLat ?? null;
    this.originLng = data.originLng ?? null;
    this.latestTimestamp = data.latestTimestamp ?? 0;
    this.nextTrackSeq = data.nextTrackSeq ?? 1;
    this.detectionBuffer = Array.isArray(data.detectionBuffer)
      ? data.detectionBuffer.map((d) => ({ ...d }))
      : [];
    this.tracks.clear();

    if (Array.isArray(data.tracks)) {
      for (const rec of data.tracks) {
        this.tracks.set(rec.trackId, {
          trackId: rec.trackId,
          uavClass: rec.uavClass,
          kalman: KalmanFilter2D.deserialize(rec.kalman),
          confidence: rec.confidence,
          snrDb: rec.snrDb,
          nodeCount: rec.nodeCount,
          contributingNodes: [...rec.contributingNodes],
          lastUpdated: rec.lastUpdated,
          bearingDeg: rec.bearingDeg ?? null,
          history: rec.history.map((h) => ({ ...h }))
        });
      }
    }
  }

  static restore(
    data: SerializedBearingFusion,
    options: BearingFusionOptions = {}
  ): BearingFusion {
    const fusion = new BearingFusion(options);
    fusion.restore(data);
    return fusion;
  }

  private upsertDetection(incoming: BufferedDetection): void {
    const existingIdx = this.detectionBuffer.findIndex(
      (d) => d.nodeId === incoming.nodeId && d.uavClass === incoming.uavClass
    );
    if (existingIdx >= 0) {
      if (incoming.timestamp >= this.detectionBuffer[existingIdx].timestamp) {
        this.detectionBuffer[existingIdx] = incoming;
      }
    } else {
      this.detectionBuffer.push(incoming);
    }
  }

  private pruneExpiredDetections(now: number): void {
    this.detectionBuffer = this.detectionBuffer.filter(
      (d) => now - d.timestamp <= this.clusterWindowMs
    );
  }

  private pruneStaleTracks(now: number): boolean {
    let removed = false;
    for (const [trackId, track] of this.tracks.entries()) {
      if (now - track.lastUpdated > this.trackStaleMs) {
        this.tracks.delete(trackId);
        removed = true;
      }
    }
    return removed;
  }

  private buildClusterForDetection(
    seedNodeId: string,
    uavClass: UavClass,
    seedTimestamp: number
  ): ClusterFusionEstimate | null {
    const candidates = this.detectionBuffer.filter(
      (d) =>
        d.uavClass === uavClass &&
        Math.abs(seedTimestamp - d.timestamp) <= this.clusterWindowMs
    );

    const seedIdx = candidates.findIndex((d) => d.nodeId === seedNodeId);
    if (seedIdx < 0) {
      return null;
    }

    // Connected-component clustering within clusterWindowMs and clusterRadiusM
    const visited = new Set<number>([seedIdx]);
    const queue = [seedIdx];
    const clusterItems: BufferedDetection[] = [];

    while (queue.length > 0) {
      const idx = queue.shift()!;
      const current = candidates[idx];
      clusterItems.push(current);

      for (let i = 0; i < candidates.length; i++) {
        if (visited.has(i)) {
          continue;
        }
        const other = candidates[i];
        const dt = Math.abs(current.timestamp - other.timestamp);
        if (dt > this.clusterWindowMs) {
          continue;
        }
        const dist = haversineDistanceMeters(
          current.lat,
          current.lng,
          other.lat,
          other.lng
        );
        if (dist <= this.clusterRadiusM) {
          visited.add(i);
          queue.push(i);
        }
      }
    }

    const originLat = this.originLat!;
    const originLng = this.originLng!;

    let weightSum = 0;
    let weightedX = 0;
    let weightedY = 0;
    let weightedConf = 0;
    let weightedAccSq = 0;
    let maxSnr = -Infinity;
    let maxTs = -Infinity;
    const nodeIds: string[] = [];
    const nodeBearingInputs: Array<{
      x: number;
      y: number;
      timestamp: number;
      weight: number;
    }> = [];

    for (const item of clusterItems) {
      const w = snrToWeight(item.snrDb);
      const local = latLngToLocalMeters(item.lat, item.lng, originLat, originLng);
      weightSum += w;
      weightedX += w * local.x;
      weightedY += w * local.y;
      weightedConf += w * item.confidence;
      const effAcc = Math.max(10, item.accuracy);
      weightedAccSq += w * effAcc * effAcc;
      if (item.snrDb > maxSnr) {
        maxSnr = item.snrDb;
      }
      if (item.timestamp > maxTs) {
        maxTs = item.timestamp;
      }
      nodeIds.push(item.nodeId);
      nodeBearingInputs.push({
        x: local.x,
        y: local.y,
        timestamp: item.timestamp,
        weight: w
      });
    }

    const centroidX = weightedX / weightSum;
    const centroidY = weightedY / weightSum;
    const { lat: centroidLat, lng: centroidLng } = localMetersToLatLng(
      centroidX,
      centroidY,
      originLat,
      originLng
    );

    const bearingDeg = computeMultiNodeBearing(
      nodeBearingInputs,
      centroidX,
      centroidY
    );

    const avgAccSq = weightedAccSq / weightSum;
    const measVarM2 = Math.max(100, avgAccSq / Math.max(1, nodeIds.length));

    return {
      uavClass,
      centroidLat,
      centroidLng,
      centroidX,
      centroidY,
      timestamp: maxTs,
      confidence: clamp(weightedConf / weightSum, 0, 1),
      snrDb: maxSnr,
      nodeCount: nodeIds.length,
      nodeIds,
      bearingDeg,
      measVarM2
    };
  }

  private associateAndUpdateTrack(cluster: ClusterFusionEstimate): Track {
    const matchedTrack = this.findAssociatedTrack(cluster);

    if (matchedTrack) {
      const dtSec = Math.max(0, (cluster.timestamp - matchedTrack.lastUpdated) / 1000);
      if (dtSec > 0) {
        matchedTrack.kalman.predict(dtSec);
      }
      matchedTrack.kalman.update(
        cluster.centroidX,
        cluster.centroidY,
        cluster.measVarM2
      );

      const state = matchedTrack.kalman.getState();
      matchedTrack.confidence = cluster.confidence;
      matchedTrack.snrDb = cluster.snrDb;
      matchedTrack.nodeCount = cluster.nodeCount;
      matchedTrack.contributingNodes = [...cluster.nodeIds];
      matchedTrack.lastUpdated = Math.max(matchedTrack.lastUpdated, cluster.timestamp);
      if (cluster.bearingDeg !== null) {
        matchedTrack.bearingDeg = cluster.bearingDeg;
      }
      matchedTrack.history.push({
        lat: state.lat,
        lng: state.lng,
        timestamp: cluster.timestamp
      });
      if (matchedTrack.history.length > this.maxHistoryPoints) {
        matchedTrack.history = matchedTrack.history.slice(-this.maxHistoryPoints);
      }

      return this.toPublicTrack(matchedTrack);
    }

    const trackId = `trk-${this.nextTrackSeq++}`;
    const kalman = new KalmanFilter2D(
      this.originLat!,
      this.originLng!,
      cluster.centroidX,
      cluster.centroidY,
      { initialPosVarM2: cluster.measVarM2 }
    );
    const state = kalman.getState();

    const newTrack: InternalTrack = {
      trackId,
      uavClass: cluster.uavClass,
      kalman,
      confidence: cluster.confidence,
      snrDb: cluster.snrDb,
      nodeCount: cluster.nodeCount,
      contributingNodes: [...cluster.nodeIds],
      lastUpdated: cluster.timestamp,
      bearingDeg: cluster.bearingDeg,
      history: [
        {
          lat: state.lat,
          lng: state.lng,
          timestamp: cluster.timestamp
        }
      ]
    };

    this.tracks.set(trackId, newTrack);
    return this.toPublicTrack(newTrack);
  }

  private findAssociatedTrack(cluster: ClusterFusionEstimate): InternalTrack | null {
    let bestTrack: InternalTrack | null = null;
    let bestDist = Infinity;

    for (const track of this.tracks.values()) {
      if (track.uavClass !== cluster.uavClass) {
        continue;
      }

      const dtSec = Math.max(0, (cluster.timestamp - track.lastUpdated) / 1000);
      const currentState = track.kalman.getLocalPosition();
      const vel = track.kalman.getState();
      const predX = currentState.x + vel.vx * dtSec;
      const predY = currentState.y + vel.vy * dtSec;
      const dist = Math.hypot(cluster.centroidX - predX, cluster.centroidY - predY);

      if (dist <= this.associationGateM && dist < bestDist) {
        bestDist = dist;
        bestTrack = track;
      }
    }

    if (bestTrack) {
      return bestTrack;
    }

    // If a new node joins an existing cluster within the 3000 ms fusion window,
    // allow the track spawned from a strict subset of that cluster's nodes to expand
    // rather than splitting a single fused target across two tracks.
    const clusterNodeSet = new Set(cluster.nodeIds);
    let bestSeedTrack: InternalTrack | null = null;
    let bestSeedDist = Infinity;

    for (const track of this.tracks.values()) {
      if (track.uavClass !== cluster.uavClass) {
        continue;
      }
      if (cluster.timestamp - track.lastUpdated > this.clusterWindowMs) {
        continue;
      }
      if (clusterNodeSet.size <= track.contributingNodes.length) {
        continue;
      }
      const isSubset = track.contributingNodes.every((n) => clusterNodeSet.has(n));
      if (!isSubset) {
        continue;
      }
      const pos = track.kalman.getLocalPosition();
      const dist = Math.hypot(cluster.centroidX - pos.x, cluster.centroidY - pos.y);
      if (dist < bestSeedDist) {
        bestSeedDist = dist;
        bestSeedTrack = track;
      }
    }

    return bestSeedTrack;
  }

  private toPublicTrack(t: InternalTrack): Track {
    const state = t.kalman.getState();
    const headingDeg =
      state.speedMps > 0.5
        ? state.headingDeg
        : t.bearingDeg !== null
          ? t.bearingDeg
          : state.headingDeg;

    return {
      trackId: t.trackId,
      class: t.uavClass,
      lat: state.lat,
      lng: state.lng,
      vx: state.vx,
      vy: state.vy,
      speedMps: state.speedMps,
      headingDeg,
      confidence: t.confidence,
      snrDb: t.snrDb,
      nodeCount: t.nodeCount,
      lastUpdated: t.lastUpdated,
      history: t.history.map((h) => ({ ...h }))
    };
  }
}
