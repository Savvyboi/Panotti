import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  buildMeshWebSocketUrl,
  getH3CellInfo,
  NodeTelemetryClient,
  shouldTransmitDetection,
  type WebSocketLike,
} from '../src/mesh/NodeTelemetry';
import type { MeshZoneState } from '../src/types';

class MockWebSocket implements WebSocketLike {
  public readyState = 0; // CONNECTING
  public sent: string[] = [];
  public onopen: ((ev: unknown) => void) | null = null;
  public onmessage: ((ev: { data: string }) => void) | null = null;
  public onerror: ((ev: unknown) => void) | null = null;
  public onclose: ((ev: { code?: number; reason?: string }) => void) | null = null;
  public readonly url: string;

  constructor(url: string) {
    this.url = url;
  }

  simulateOpen(): void {
    this.readyState = 1;
    this.onopen?.({});
  }

  simulateMessage(payload: unknown): void {
    this.onmessage?.({ data: JSON.stringify(payload) });
  }

  simulateClose(code = 1006, reason = 'drop'): void {
    this.readyState = 3;
    this.onclose?.({ code, reason });
  }

  send(data: string): void {
    this.sent.push(data);
  }

  close(code = 1000, reason = 'closed'): void {
    this.readyState = 3;
    this.onclose?.({ code, reason });
  }
}

describe('NodeTelemetry', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('computes H3 Resolution 6 cell and constructs valid WebSocket URL', () => {
    const info = getH3CellInfo(50.4501, 30.5234);
    expect(info.cellId).toMatch(/^86[0-9a-f]+$/i);
    expect(info.boundary.length).toBeGreaterThanOrEqual(6);

    const url = buildMeshWebSocketUrl('http://localhost:8787/ws', 50.4501, 30.5234, 'node-alpha');
    expect(url).toBe(
      'ws://localhost:8787/ws?lat=50.4501&lng=30.5234&nodeId=node-alpha'
    );
  });

  it('enforces >= 0.65 confidence and UAV-only transmission gating', () => {
    expect(shouldTransmitDetection('Shahed_Piston_UAV', 0.65)).toBe(true);
    expect(shouldTransmitDetection('Multirotor_UAV', 0.88)).toBe(true);
    expect(shouldTransmitDetection('FixedWing_Electric_UAV', 0.7)).toBe(true);
    expect(shouldTransmitDetection('Shahed_Piston_UAV', 0.649)).toBe(false);
    expect(shouldTransmitDetection('Ground_Combustion_Noise', 0.99)).toBe(false);
    expect(shouldTransmitDetection('Environmental_Background', 0.99)).toBe(false);
  });

  it('transmits exact detection JSON and 10s heartbeat, and receives zone snapshots', () => {
    const sockets: MockWebSocket[] = [];
    const zoneUpdates: MeshZoneState[] = [];

    const client = new NodeTelemetryClient({
      nodeId: 'node-test-01',
      baseUrl: 'ws://localhost:8787/ws',
      location: { lat: 50.4501, lng: 30.5234, accuracy: 18 },
      webSocketFactory: (url) => {
        const ws = new MockWebSocket(url);
        sockets.push(ws);
        return ws;
      },
      onZoneUpdate: (zone) => {
        zoneUpdates.push(zone);
      },
    });

    client.connect();
    expect(sockets).toHaveLength(1);
    sockets[0].simulateOpen();

    // Initial heartbeat on connect
    expect(sockets[0].sent).toHaveLength(1);
    const hb0 = JSON.parse(sockets[0].sent[0]);
    expect(hb0.type).toBe('heartbeat');
    expect(hb0.nodeId).toBe('node-test-01');
    expect(hb0.location).toEqual({ lat: 50.4501, lng: 30.5234, accuracy: 18 });

    // Sub-threshold UAV suppressed
    const suppressedLow = client.reportDetection({
      class: 'Shahed_Piston_UAV',
      confidence: 0.64,
      snr_db: 12.1,
      peak_freq_hz: 92.5,
    });
    expect(suppressedLow.transmitted).toBe(false);
    expect(sockets[0].sent).toHaveLength(1);

    // Non-UAV class suppressed even at 0.99
    const suppressedNoise = client.reportDetection({
      class: 'Ground_Combustion_Noise',
      confidence: 0.99,
      snr_db: 22.0,
      peak_freq_hz: 110.0,
    });
    expect(suppressedNoise.transmitted).toBe(false);
    expect(sockets[0].sent).toHaveLength(1);

    // Valid UAV >= 0.65 transmitted with exact schema
    const tx = client.reportDetection(
      {
        class: 'Shahed_Piston_UAV',
        confidence: 0.91,
        snr_db: 18.4,
        peak_freq_hz: 92.5,
      },
      1700000000123
    );
    expect(tx.transmitted).toBe(true);
    expect(sockets[0].sent).toHaveLength(2);

    const sentDet = JSON.parse(sockets[0].sent[1]);
    expect(sentDet).toEqual({
      nodeId: 'node-test-01',
      timestamp: 1700000000123,
      location: { lat: 50.4501, lng: 30.5234, accuracy: 18 },
      detection: {
        class: 'Shahed_Piston_UAV',
        confidence: 0.91,
        snr_db: 18.4,
        peak_freq_hz: 92.5,
      },
    });

    // Advance 10s -> heartbeat sent
    vi.advanceTimersByTime(10000);
    expect(sockets[0].sent).toHaveLength(3);
    expect(JSON.parse(sockets[0].sent[2]).type).toBe('heartbeat');

    // Receive snapshot & track_update
    sockets[0].simulateMessage({
      type: 'snapshot',
      cellId: client.getCellId(),
      boundary: [[50.45, 30.52], [50.46, 30.52], [50.46, 30.53]],
      tracks: [
        {
          trackId: 'trk-1',
          class: 'Shahed_Piston_UAV',
          lat: 50.452,
          lng: 30.528,
          vx: 40,
          vy: 10,
          speedMps: 41.2,
          headingDeg: 76,
          confidence: 0.91,
          snrDb: 18.4,
          nodeCount: 2,
          lastUpdated: 1700000001000,
          history: [{ lat: 50.451, lng: 30.526, timestamp: 1700000000000 }],
        },
      ],
      nodes: [{ nodeId: 'node-peer-2', lat: 50.455, lng: 30.53 }],
    });

    expect(zoneUpdates).toHaveLength(1);
    expect(zoneUpdates[0].tracks[0].trackId).toBe('trk-1');
    expect(zoneUpdates[0].nodes[0].nodeId).toBe('node-peer-2');

    // Unexpected close triggers exponential reconnect backoff
    sockets[0].simulateClose();
    expect(client.getState()).toBe('reconnecting');
    vi.advanceTimersByTime(1000);
    expect(sockets).toHaveLength(2);

    client.disconnect();
    expect(client.getState()).toBe('disconnected');
  });
});
