export const DETECTION_CLASSES = [
  "Shahed_Piston_UAV",
  "Multirotor_UAV",
  "FixedWing_Electric_UAV",
  "Ground_Combustion_Noise",
  "Environmental_Background"
] as const;

export type DetectionClass = (typeof DETECTION_CLASSES)[number];

export const UAV_CLASSES = [
  "Shahed_Piston_UAV",
  "Multirotor_UAV",
  "FixedWing_Electric_UAV"
] as const;

export type UavClass = (typeof UAV_CLASSES)[number];

export interface GeoLocation {
  lat: number;
  lng: number;
  accuracy: number;
}

export interface HeartbeatLocation {
  lat: number;
  lng: number;
  accuracy?: number;
}

export interface DetectionDetails {
  class: DetectionClass;
  confidence: number;
  snr_db: number;
  peak_freq_hz: number;
}

export interface DetectionPayload {
  type?: "detection";
  nodeId: string;
  timestamp: number;
  location: GeoLocation;
  detection: DetectionDetails;
}

export interface NormalizedDetectionMessage {
  type: "detection";
  nodeId: string;
  timestamp: number;
  location: GeoLocation;
  detection: DetectionDetails;
}

export interface HeartbeatPayload {
  type: "heartbeat";
  nodeId: string;
  timestamp: number;
  location: HeartbeatLocation;
}

export type ClientMessage = NormalizedDetectionMessage | HeartbeatPayload;

export type ParseClientMessageResult =
  | { ok: true; message: ClientMessage }
  | { ok: false; error: string };

export interface TrackHistoryPoint {
  lat: number;
  lng: number;
  timestamp: number;
}

export interface Track {
  trackId: string;
  class: DetectionClass;
  lat: number;
  lng: number;
  vx: number;
  vy: number;
  speedMps: number;
  headingDeg: number;
  confidence: number;
  snrDb: number;
  nodeCount: number;
  lastUpdated: number;
  history: TrackHistoryPoint[];
}

export interface NodeState {
  nodeId: string;
  lat: number;
  lng: number;
  accuracy?: number;
  lastSeen: number;
}

export interface SnapshotMessage {
  type: "snapshot";
  cellId: string;
  boundary: Array<[number, number]>;
  tracks: Track[];
  nodes: NodeState[];
}

export interface TrackUpdateMessage {
  type: "track_update";
  cellId: string;
  timestamp: number;
  boundary: Array<[number, number]>;
  tracks: Track[];
  nodes: NodeState[];
}

export type ServerBroadcastMessage = SnapshotMessage | TrackUpdateMessage;

export interface Env {
  SPATIAL_ZONE: DurableObjectNamespace;
  ALLOWED_ORIGINS?: string;
}

const DETECTION_CLASS_SET = new Set<string>(DETECTION_CLASSES);
const UAV_CLASS_SET = new Set<string>(UAV_CLASSES);

export function isDetectionClass(value: unknown): value is DetectionClass {
  return typeof value === "string" && DETECTION_CLASS_SET.has(value);
}

export function isUavClass(value: unknown): value is UavClass {
  return typeof value === "string" && UAV_CLASS_SET.has(value);
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function validateLatLng(
  loc: Record<string, unknown>
): { ok: true; lat: number; lng: number } | { ok: false; error: string } {
  if (!isFiniteNumber(loc.lat) || loc.lat < -90 || loc.lat > 90) {
    return { ok: false, error: "location.lat must be a finite number in [-90, 90]" };
  }
  if (!isFiniteNumber(loc.lng) || loc.lng < -180 || loc.lng > 180) {
    return { ok: false, error: "location.lng must be a finite number in [-180, 180]" };
  }
  return { ok: true, lat: loc.lat, lng: loc.lng };
}

/**
 * Runtime parser and bounds validator for incoming WebSocket / HTTP telemetry payloads.
 * Accepts either raw JSON strings or pre-parsed objects.
 */
export function parseClientMessage(raw: unknown): ParseClientMessageResult {
  let parsed: unknown = raw;
  if (typeof raw === "string") {
    try {
      parsed = JSON.parse(raw);
    } catch {
      return { ok: false, error: "Invalid JSON payload" };
    }
  }

  if (!isPlainObject(parsed)) {
    return { ok: false, error: "Payload must be a JSON object" };
  }

  if (typeof parsed.nodeId !== "string" || parsed.nodeId.trim().length === 0) {
    return { ok: false, error: "nodeId must be a non-empty string" };
  }
  const nodeId = parsed.nodeId.trim();
  if (nodeId.length > 128) {
    return { ok: false, error: "nodeId must not exceed 128 characters" };
  }

  if (!isFiniteNumber(parsed.timestamp) || parsed.timestamp <= 0) {
    return { ok: false, error: "timestamp must be a positive finite number" };
  }

  if (!isPlainObject(parsed.location)) {
    return { ok: false, error: "location must be an object" };
  }

  const coords = validateLatLng(parsed.location);
  if (!coords.ok) {
    return coords;
  }

  const rawAccuracy = parsed.location.accuracy;
  if (rawAccuracy !== undefined) {
    if (!isFiniteNumber(rawAccuracy) || rawAccuracy < 0 || rawAccuracy > 1_000_000) {
      return {
        ok: false,
        error: "location.accuracy must be a finite number in [0, 1000000]"
      };
    }
  }

  const msgType = parsed.type;

  if (msgType === "heartbeat") {
    return {
      ok: true,
      message: {
        type: "heartbeat",
        nodeId,
        timestamp: parsed.timestamp,
        location: {
          lat: coords.lat,
          lng: coords.lng,
          ...(rawAccuracy !== undefined ? { accuracy: rawAccuracy } : {})
        }
      }
    };
  }

  if (msgType !== undefined && msgType !== "detection") {
    return { ok: false, error: `Unsupported message type: ${String(msgType)}` };
  }

  if (!isPlainObject(parsed.detection)) {
    return { ok: false, error: "detection must be an object" };
  }

  const det = parsed.detection;
  if (!isDetectionClass(det.class)) {
    return {
      ok: false,
      error: `detection.class must be one of: ${DETECTION_CLASSES.join(", ")}`
    };
  }

  if (!isFiniteNumber(det.confidence) || det.confidence < 0 || det.confidence > 1) {
    return {
      ok: false,
      error: "detection.confidence must be a finite number in [0, 1]"
    };
  }

  if (!isFiniteNumber(det.snr_db) || det.snr_db < -100 || det.snr_db > 120) {
    return {
      ok: false,
      error: "detection.snr_db must be a finite number in [-100, 120]"
    };
  }

  if (!isFiniteNumber(det.peak_freq_hz) || det.peak_freq_hz < 0 || det.peak_freq_hz > 24000) {
    return {
      ok: false,
      error: "detection.peak_freq_hz must be a finite number in [0, 24000]"
    };
  }

  return {
    ok: true,
    message: {
      type: "detection",
      nodeId,
      timestamp: parsed.timestamp,
      location: {
        lat: coords.lat,
        lng: coords.lng,
        accuracy: rawAccuracy ?? 10
      },
      detection: {
        class: det.class,
        confidence: det.confidence,
        snr_db: det.snr_db,
        peak_freq_hz: det.peak_freq_hz
      }
    }
  };
}
