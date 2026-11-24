export const UAV_CLASSES = [
  'Shahed_Piston_UAV',
  'Multirotor_UAV',
  'FixedWing_Electric_UAV',
  'Ground_Combustion_Noise',
  'Environmental_Background',
] as const;

export type AudioClass = (typeof UAV_CLASSES)[number];

export const TRANSMITTABLE_UAV_CLASSES: ReadonlySet<AudioClass> = new Set<AudioClass>([
  'Shahed_Piston_UAV',
  'Multirotor_UAV',
  'FixedWing_Electric_UAV',
]);

export const CONFIDENCE_THRESHOLD = 0.65;

export type SimulationProfile = 'shahed' | 'multirotor' | 'background';

export type ExecutionProviderUsed = 'webgpu' | 'wasm';

export type EngineStatus = 'uninitialized' | 'loading' | 'ready' | 'unavailable' | 'error';

export type AudioCaptureState = 'idle' | 'microphone' | 'simulation';

export type MeshConnectionState = 'disconnected' | 'connecting' | 'connected' | 'reconnecting';

export interface SignalMetrics {
  snr_db: number;
  peak_freq_hz: number;
}

export interface InferenceResult {
  topClass: AudioClass;
  confidence: number;
  probabilities: Record<AudioClass, number>;
  logits: number[];
  provider: ExecutionProviderUsed;
  inferenceMs: number;
}

export interface NodeLocation {
  lat: number;
  lng: number;
  accuracy: number;
}

export interface DetectionPayload {
  nodeId: string;
  timestamp: number;
  location: NodeLocation;
  detection: {
    class: AudioClass;
    confidence: number;
    snr_db: number;
    peak_freq_hz: number;
  };
}

export interface HeartbeatPayload {
  type: 'heartbeat';
  nodeId: string;
  timestamp: number;
  location: NodeLocation;
}

export interface TrackPoint {
  lat: number;
  lng: number;
  timestamp: number;
}

export interface TrackState {
  trackId: string;
  class: AudioClass;
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
  history: TrackPoint[];
}

export interface MeshNodeInfo {
  nodeId: string;
  lat: number;
  lng: number;
  accuracy?: number;
  lastSeen?: number;
}

export interface MeshZoneState {
  cellId: string;
  boundary: [number, number][];
  tracks: TrackState[];
  nodes: MeshNodeInfo[];
  timestamp?: number;
}
