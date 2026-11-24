import { cellToBoundary, latLngToCell } from 'h3-js';
import {
  CONFIDENCE_THRESHOLD,
  TRANSMITTABLE_UAV_CLASSES,
  UAV_CLASSES,
  type AudioClass,
  type DetectionPayload,
  type HeartbeatPayload,
  type MeshConnectionState,
  type MeshNodeInfo,
  type MeshZoneState,
  type NodeLocation,
  type TrackState,
} from '../types';

export const H3_RESOLUTION = 6;
export const HEARTBEAT_INTERVAL_MS = 10000;

export function getDefaultMeshWsUrl(): string {
  const envUrl =
    typeof import.meta !== 'undefined' &&
    typeof import.meta.env !== 'undefined' &&
    import.meta.env.VITE_MESH_WS_URL
      ? String(import.meta.env.VITE_MESH_WS_URL)
      : '';
  return envUrl || 'ws://localhost:8787/ws';
}

export function getH3CellInfo(lat: number, lng: number): {
  cellId: string;
  boundary: [number, number][];
} {
  const cellId = latLngToCell(lat, lng, H3_RESOLUTION);
  const boundary = cellToBoundary(cellId) as [number, number][];
  return { cellId, boundary };
}

export function buildMeshWebSocketUrl(
  baseUrl: string,
  lat: number,
  lng: number,
  nodeId: string
): string {
  const normalizedBase = baseUrl.trim().replace(/^http:/i, 'ws:').replace(/^https:/i, 'wss:');
  const url = new URL(normalizedBase);
  url.searchParams.set('lat', String(lat));
  url.searchParams.set('lng', String(lng));
  url.searchParams.set('nodeId', nodeId);
  return url.toString();
}

export function shouldTransmitDetection(detectionClass: AudioClass, confidence: number): boolean {
  return (
    TRANSMITTABLE_UAV_CLASSES.has(detectionClass) &&
    Number.isFinite(confidence) &&
    confidence >= CONFIDENCE_THRESHOLD
  );
}

export function buildDetectionPayload(
  nodeId: string,
  location: NodeLocation,
  detection: {
    class: AudioClass;
    confidence: number;
    snr_db: number;
    peak_freq_hz: number;
  },
  timestamp = Date.now()
): DetectionPayload {
  return {
    nodeId,
    timestamp,
    location: {
      lat: location.lat,
      lng: location.lng,
      accuracy: location.accuracy,
    },
    detection: {
      class: detection.class,
      confidence: detection.confidence,
      snr_db: detection.snr_db,
      peak_freq_hz: detection.peak_freq_hz,
    },
  };
}

export function buildHeartbeatPayload(
  nodeId: string,
  location: NodeLocation,
  timestamp = Date.now()
): HeartbeatPayload {
  return {
    type: 'heartbeat',
    nodeId,
    timestamp,
    location: {
      lat: location.lat,
      lng: location.lng,
      accuracy: location.accuracy,
    },
  };
}

export interface WebSocketLike {
  readonly readyState: number;
  send(data: string): void;
  close(code?: number, reason?: string): void;
  onopen: ((ev: unknown) => void) | null;
  onmessage: ((ev: { data: string }) => void) | null;
  onerror: ((ev: unknown) => void) | null;
  onclose: ((ev: { code?: number; reason?: string }) => void) | null;
}

export interface NodeTelemetryOptions {
  nodeId: string;
  baseUrl?: string;
  location: NodeLocation;
  heartbeatIntervalMs?: number;
  reconnectBaseMs?: number;
  reconnectMaxMs?: number;
  webSocketFactory?: (url: string) => WebSocketLike;
  onStateChange?: (state: MeshConnectionState, detail?: string) => void;
  onZoneUpdate?: (zone: MeshZoneState) => void;
  onLog?: (level: 'info' | 'warn' | 'tx', message: string) => void;
}

export interface DetectionDispatchResult {
  transmitted: boolean;
  payload?: DetectionPayload;
  reason?: string;
}

const WS_OPEN = 1;

function normalizeAudioClass(raw: unknown): AudioClass {
  if (typeof raw === 'string' && (UAV_CLASSES as readonly string[]).includes(raw)) {
    return raw as AudioClass;
  }
  return 'Shahed_Piston_UAV';
}

function normalizeNodes(rawNodes: unknown): MeshNodeInfo[] {
  if (!Array.isArray(rawNodes)) return [];
  const parsed: MeshNodeInfo[] = [];

  for (const item of rawNodes) {
    if (!item || typeof item !== 'object') continue;
    const obj = item as Record<string, unknown>;
    const nodeId = typeof obj.nodeId === 'string' ? obj.nodeId : 'unknown-node';

    let lat: number | undefined;
    let lng: number | undefined;
    let accuracy: number | undefined;

    if (typeof obj.lat === 'number' && typeof obj.lng === 'number') {
      lat = obj.lat;
      lng = obj.lng;
      accuracy = typeof obj.accuracy === 'number' ? obj.accuracy : undefined;
    } else if (obj.location && typeof obj.location === 'object') {
      const loc = obj.location as Record<string, unknown>;
      if (typeof loc.lat === 'number' && typeof loc.lng === 'number') {
        lat = loc.lat;
        lng = loc.lng;
        accuracy = typeof loc.accuracy === 'number' ? loc.accuracy : undefined;
      }
    }

    if (lat !== undefined && lng !== undefined && Number.isFinite(lat) && Number.isFinite(lng)) {
      const lastSeen =
        typeof obj.lastSeen === 'number'
          ? obj.lastSeen
          : typeof obj.lastHeartbeat === 'number'
            ? obj.lastHeartbeat
            : undefined;
      parsed.push({ nodeId, lat, lng, accuracy, lastSeen });
    }
  }

  return parsed;
}

function normalizeTracks(rawTracks: unknown): TrackState[] {
  if (!Array.isArray(rawTracks)) return [];
  const parsed: TrackState[] = [];

  for (const item of rawTracks) {
    if (!item || typeof item !== 'object') continue;
    const t = item as Record<string, unknown>;
    if (typeof t.trackId !== 'string' || typeof t.lat !== 'number' || typeof t.lng !== 'number') {
      continue;
    }

    const history = Array.isArray(t.history)
      ? t.history
          .filter(
            (pt): pt is { lat: number; lng: number; timestamp: number } =>
              Boolean(pt) &&
              typeof pt === 'object' &&
              typeof (pt as { lat?: unknown }).lat === 'number' &&
              typeof (pt as { lng?: unknown }).lng === 'number' &&
              typeof (pt as { timestamp?: unknown }).timestamp === 'number'
          )
          .map((pt) => ({ lat: pt.lat, lng: pt.lng, timestamp: pt.timestamp }))
      : [];

    parsed.push({
      trackId: t.trackId,
      class: normalizeAudioClass(t.class),
      lat: t.lat,
      lng: t.lng,
      vx: typeof t.vx === 'number' ? t.vx : 0,
      vy: typeof t.vy === 'number' ? t.vy : 0,
      speedMps: typeof t.speedMps === 'number' ? t.speedMps : 0,
      headingDeg: typeof t.headingDeg === 'number' ? t.headingDeg : 0,
      confidence: typeof t.confidence === 'number' ? t.confidence : 0,
      snrDb: typeof t.snrDb === 'number' ? t.snrDb : 0,
      nodeCount: typeof t.nodeCount === 'number' ? t.nodeCount : 1,
      lastUpdated: typeof t.lastUpdated === 'number' ? t.lastUpdated : Date.now(),
      history,
    });
  }

  return parsed;
}

export class NodeTelemetryClient {
  public readonly nodeId: string;

  private baseUrl: string;
  private location: NodeLocation;
  private currentCellId: string;

  private readonly heartbeatIntervalMs: number;
  private readonly reconnectBaseMs: number;
  private readonly reconnectMaxMs: number;
  private readonly webSocketFactory: (url: string) => WebSocketLike;
  private readonly onStateChange?: (state: MeshConnectionState, detail?: string) => void;
  private readonly onZoneUpdate?: (zone: MeshZoneState) => void;
  private readonly onLog?: (level: 'info' | 'warn' | 'tx', message: string) => void;

  private ws: WebSocketLike | null = null;
  private state: MeshConnectionState = 'disconnected';
  private shouldReconnect = false;
  private reconnectAttempts = 0;
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(options: NodeTelemetryOptions) {
    this.nodeId = options.nodeId;
    this.baseUrl = options.baseUrl ?? getDefaultMeshWsUrl();
    this.location = { ...options.location };
    this.currentCellId = latLngToCell(this.location.lat, this.location.lng, H3_RESOLUTION);
    this.heartbeatIntervalMs = options.heartbeatIntervalMs ?? HEARTBEAT_INTERVAL_MS;
    this.reconnectBaseMs = options.reconnectBaseMs ?? 1000;
    this.reconnectMaxMs = options.reconnectMaxMs ?? 15000;
    this.webSocketFactory =
      options.webSocketFactory ??
      ((url: string) => new WebSocket(url) as unknown as WebSocketLike);
    this.onStateChange = options.onStateChange;
    this.onZoneUpdate = options.onZoneUpdate;
    this.onLog = options.onLog;
  }

  public getState(): MeshConnectionState {
    return this.state;
  }

  public getCellId(): string {
    return this.currentCellId;
  }

  public getLocation(): NodeLocation {
    return { ...this.location };
  }

  public connect(baseUrl?: string, location?: NodeLocation): void {
    if (baseUrl !== undefined) {
      this.baseUrl = baseUrl;
    }
    if (location !== undefined) {
      this.location = { ...location };
      this.currentCellId = latLngToCell(this.location.lat, this.location.lng, H3_RESOLUTION);
    }

    this.clearTimers();
    this.closeCurrentSocket();
    this.shouldReconnect = true;
    this.reconnectAttempts = 0;
    this.openSocket();
  }

  public disconnect(): void {
    this.shouldReconnect = false;
    this.clearTimers();
    this.closeCurrentSocket();
    this.setState('disconnected', 'Disconnected');
  }

  public updateLocation(location: NodeLocation): void {
    this.location = { ...location };
    const nextCell = latLngToCell(location.lat, location.lng, H3_RESOLUTION);
    const cellChanged = nextCell !== this.currentCellId;
    this.currentCellId = nextCell;

    if (cellChanged && this.shouldReconnect) {
      this.onLog?.('info', `H3 zone changed to ${nextCell}; re-anchoring mesh WebSocket`);
      this.connect(this.baseUrl, this.location);
      return;
    }

    if (this.ws && this.ws.readyState === WS_OPEN) {
      this.sendHeartbeat();
    }
  }

  public sendHeartbeat(timestamp = Date.now()): boolean {
    if (!this.ws || this.ws.readyState !== WS_OPEN) {
      return false;
    }
    const payload = buildHeartbeatPayload(this.nodeId, this.location, timestamp);
    this.ws.send(JSON.stringify(payload));
    return true;
  }

  public reportDetection(
    detection: {
      class: AudioClass;
      confidence: number;
      snr_db: number;
      peak_freq_hz: number;
    },
    timestamp = Date.now()
  ): DetectionDispatchResult {
    if (!shouldTransmitDetection(detection.class, detection.confidence)) {
      const reason = !TRANSMITTABLE_UAV_CLASSES.has(detection.class)
        ? `Non-UAV class (${detection.class}) suppressed locally`
        : `Confidence ${(detection.confidence * 100).toFixed(1)}% below ${(CONFIDENCE_THRESHOLD * 100).toFixed(0)}% threshold`;
      return { transmitted: false, reason };
    }

    const payload = buildDetectionPayload(this.nodeId, this.location, detection, timestamp);

    if (!this.ws || this.ws.readyState !== WS_OPEN) {
      return {
        transmitted: false,
        payload,
        reason: 'Mesh WebSocket disconnected; alert held locally',
      };
    }

    this.ws.send(JSON.stringify(payload));
    return { transmitted: true, payload };
  }

  private openSocket(): void {
    let wsUrl: string;
    try {
      wsUrl = buildMeshWebSocketUrl(
        this.baseUrl,
        this.location.lat,
        this.location.lng,
        this.nodeId
      );
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Invalid mesh WebSocket URL';
      this.setState('disconnected', msg);
      this.onLog?.('warn', msg);
      this.shouldReconnect = false;
      return;
    }

    this.setState(this.reconnectAttempts > 0 ? 'reconnecting' : 'connecting', wsUrl);

    let socket: WebSocketLike;
    try {
      socket = this.webSocketFactory(wsUrl);
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'WebSocket creation failed';
      this.onLog?.('warn', msg);
      this.scheduleReconnect();
      return;
    }

    this.ws = socket;

    socket.onopen = () => {
      if (this.ws !== socket) return;
      this.reconnectAttempts = 0;
      this.setState('connected', `Cell ${this.currentCellId}`);
      this.onLog?.('info', `Mesh connected (${this.currentCellId})`);
      this.sendHeartbeat();
      this.startHeartbeatLoop();
    };

    socket.onmessage = (event: { data: string }) => {
      if (this.ws !== socket) return;
      this.handleServerMessage(event.data);
    };

    socket.onerror = () => {
      if (this.ws !== socket) return;
      this.onLog?.('warn', 'Mesh WebSocket transport error');
    };

    socket.onclose = () => {
      if (this.ws !== socket) return;
      this.ws = null;
      this.stopHeartbeatLoop();
      if (this.shouldReconnect) {
        this.scheduleReconnect();
      } else {
        this.setState('disconnected', 'Closed');
      }
    };
  }

  private handleServerMessage(rawData: string): void {
    try {
      const msg = JSON.parse(rawData) as Record<string, unknown>;
      if (msg.type !== 'snapshot' && msg.type !== 'track_update') {
        return;
      }

      const cellId = typeof msg.cellId === 'string' ? msg.cellId : this.currentCellId;
      const boundary = Array.isArray(msg.boundary)
        ? (msg.boundary as [number, number][])
        : (cellToBoundary(cellId) as [number, number][]);

      const zone: MeshZoneState = {
        cellId,
        boundary,
        tracks: normalizeTracks(msg.tracks),
        nodes: normalizeNodes(msg.nodes),
        timestamp: typeof msg.timestamp === 'number' ? msg.timestamp : undefined,
      };

      this.onZoneUpdate?.(zone);
    } catch {
      this.onLog?.('warn', 'Ignored malformed mesh message');
    }
  }

  private startHeartbeatLoop(): void {
    this.stopHeartbeatLoop();
    this.heartbeatTimer = setInterval(() => {
      this.sendHeartbeat();
    }, this.heartbeatIntervalMs);
  }

  private stopHeartbeatLoop(): void {
    if (this.heartbeatTimer !== null) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private scheduleReconnect(): void {
    if (!this.shouldReconnect || this.reconnectTimer !== null) return;
    const delay = Math.min(
      this.reconnectMaxMs,
      this.reconnectBaseMs * Math.pow(2, this.reconnectAttempts)
    );
    this.reconnectAttempts += 1;
    this.setState('reconnecting', `Retry in ${Math.round(delay / 1000)}s`);

    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      if (this.shouldReconnect) {
        this.openSocket();
      }
    }, delay);
  }

  private clearTimers(): void {
    this.stopHeartbeatLoop();
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  private closeCurrentSocket(): void {
    if (!this.ws) return;
    const oldSocket = this.ws;
    this.ws = null;
    oldSocket.onopen = null;
    oldSocket.onmessage = null;
    oldSocket.onerror = null;
    oldSocket.onclose = null;
    try {
      oldSocket.close(1000, 'Client reconnect or stop');
    } catch {
      // Ignore already-closed errors
    }
  }

  private setState(state: MeshConnectionState, detail?: string): void {
    this.state = state;
    this.onStateChange?.(state, detail);
  }
}
