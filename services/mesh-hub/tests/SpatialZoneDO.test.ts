import { latLngToCell } from "h3-js";
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { SpatialZoneDO } from "../src/SpatialZoneDO";
import type {
  Env,
  SnapshotMessage,
  TrackUpdateMessage
} from "../src/types";

class MockWebSocket {
  readonly sent: string[] = [];
  closed = false;
  closeCode?: number;
  closeReason?: string;
  private attachment: unknown = null;

  send(data: string | ArrayBuffer | ArrayBufferView): void {
    if (typeof data === "string") {
      this.sent.push(data);
    } else if (data instanceof ArrayBuffer) {
      this.sent.push(new TextDecoder().decode(data));
    } else {
      const view = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
      this.sent.push(new TextDecoder().decode(view));
    }
  }

  close(code?: number, reason?: string): void {
    this.closed = true;
    this.closeCode = code;
    this.closeReason = reason;
  }

  serializeAttachment(attachment: unknown): void {
    this.attachment = structuredClone(attachment);
  }

  deserializeAttachment(): unknown {
    return structuredClone(this.attachment);
  }
}

class MockStorage {
  readonly store = new Map<string, unknown>();
  alarmAt: number | null = null;

  async get<T = unknown>(key: string): Promise<T | undefined> {
    const val = this.store.get(key);
    return val === undefined ? undefined : (structuredClone(val) as T);
  }

  async put<T>(
    keyOrEntries: string | Record<string, T>,
    value?: T
  ): Promise<void> {
    if (typeof keyOrEntries === "string") {
      this.store.set(keyOrEntries, structuredClone(value));
    } else {
      for (const [k, v] of Object.entries(keyOrEntries)) {
        this.store.set(k, structuredClone(v));
      }
    }
  }

  async delete(key: string): Promise<boolean> {
    return this.store.delete(key);
  }

  async getAlarm(): Promise<number | null> {
    return this.alarmAt;
  }

  async setAlarm(scheduledTime: number | Date): Promise<void> {
    this.alarmAt =
      scheduledTime instanceof Date ? scheduledTime.getTime() : scheduledTime;
  }

  async deleteAlarm(): Promise<void> {
    this.alarmAt = null;
  }
}

class MockDurableObjectState {
  readonly storage = new MockStorage();
  readonly acceptedSockets: MockWebSocket[] = [];

  async blockConcurrencyWhile<T>(callback: () => Promise<T>): Promise<T> {
    return callback();
  }

  acceptWebSocket(ws: WebSocket): void {
    this.acceptedSockets.push(ws as unknown as MockWebSocket);
  }

  getWebSockets(): WebSocket[] {
    return this.acceptedSockets.filter((s) => !s.closed) as unknown as WebSocket[];
  }
}

const OriginalResponse = globalThis.Response;
const OriginalWebSocketPair = (
  globalThis as unknown as { WebSocketPair?: unknown }
).WebSocketPair;

let lastCreatedPairs: Array<{ client: MockWebSocket; server: MockWebSocket }> = [];

beforeAll(() => {
  (globalThis as unknown as { WebSocketPair: unknown }).WebSocketPair = class {
    0: MockWebSocket;
    1: MockWebSocket;
    constructor() {
      const client = new MockWebSocket();
      const server = new MockWebSocket();
      this[0] = client;
      this[1] = server;
      lastCreatedPairs.push({ client, server });
    }
  };

  // Node's global Response rejects status 101; wrap 101 responses for DO upgrade tests.
  (globalThis as unknown as { Response: unknown }).Response = class extends OriginalResponse {
    override readonly webSocket: WebSocket | null = null;
    constructor(
      body?: BodyInit | null,
      init?: ResponseInit & { webSocket?: WebSocket }
    ) {
      if (init?.status === 101) {
        super(null, { status: 200, headers: init.headers });
        Object.defineProperty(this, "status", { value: 101 });
        this.webSocket = init.webSocket ?? null;
      } else {
        super(body, init);
      }
    }
  };
});

afterEach(() => {
  lastCreatedPairs = [];
  vi.useRealTimers();
});

afterAll(() => {
  globalThis.Response = OriginalResponse;
  if (OriginalWebSocketPair === undefined) {
    delete (globalThis as unknown as { WebSocketPair?: unknown }).WebSocketPair;
  } else {
    (globalThis as unknown as { WebSocketPair: unknown }).WebSocketPair =
      OriginalWebSocketPair;
  }
});

const TEST_ENV: Env = {
  SPATIAL_ZONE: {} as DurableObjectNamespace,
  ALLOWED_ORIGINS: "http://localhost:5173"
};

describe("SpatialZoneDO", () => {
  const cellId = latLngToCell(50.45, 30.52, 6);

  it("upgrades WebSockets via Hibernation API, attaches metadata, and sends initial snapshot", async () => {
    const state = new MockDurableObjectState();
    const doInstance = new SpatialZoneDO(
      state as unknown as DurableObjectState,
      TEST_ENV
    );

    const res = await doInstance.fetch(
      new Request(`https://mesh.local/ws?lat=50.45&lng=30.52&nodeId=node-1`, {
        headers: {
          Upgrade: "websocket",
          "x-cell-id": cellId,
          "x-node-id": "node-1"
        }
      })
    );

    expect(res.status).toBe(101);
    expect(state.acceptedSockets).toHaveLength(1);
    const serverSocket = state.acceptedSockets[0];
    expect(serverSocket.deserializeAttachment()).toEqual({
      nodeId: "node-1",
      cellId
    });

    expect(serverSocket.sent).toHaveLength(1);
    const snapshot = JSON.parse(serverSocket.sent[0]) as SnapshotMessage;
    expect(snapshot.type).toBe("snapshot");
    expect(snapshot.cellId).toBe(cellId);
    expect(snapshot.boundary).toHaveLength(6);
    expect(snapshot.tracks).toEqual([]);
    expect(snapshot.nodes).toEqual([]);
  });

  it("enforces confidence >= 0.65, registers heartbeats, runs BearingFusion, persists & broadcasts", async () => {
    const state = new MockDurableObjectState();
    const doInstance = new SpatialZoneDO(
      state as unknown as DurableObjectState,
      TEST_ENV
    );

    await doInstance.fetch(
      new Request("https://mesh.local/ws", {
        headers: {
          Upgrade: "websocket",
          "x-cell-id": cellId,
          "x-node-id": "observer-1"
        }
      })
    );
    const ws = state.acceptedSockets[0];
    const now = Date.now();

    // Heartbeat registers node & broadcasts track_update
    await doInstance.webSocketMessage(
      ws as unknown as WebSocket,
      JSON.stringify({
        type: "heartbeat",
        nodeId: "node-a",
        timestamp: now,
        location: { lat: 50.45, lng: 30.52, accuracy: 10 }
      })
    );
    expect(ws.sent).toHaveLength(2);
    const hbBroadcast = JSON.parse(ws.sent[1]) as TrackUpdateMessage;
    expect(hbBroadcast.type).toBe("track_update");
    expect(hbBroadcast.nodes).toHaveLength(1);
    expect(hbBroadcast.nodes[0].nodeId).toBe("node-a");

    // Detection with confidence < 0.65 is dropped without creating track or broadcasting
    await doInstance.webSocketMessage(
      ws as unknown as WebSocket,
      JSON.stringify({
        nodeId: "node-a",
        timestamp: now + 200,
        location: { lat: 50.45, lng: 30.52, accuracy: 10 },
        detection: {
          class: "Shahed_Piston_UAV",
          confidence: 0.64,
          snr_db: 15,
          peak_freq_hz: 92.5
        }
      })
    );
    expect(ws.sent).toHaveLength(2);

    // Detection with confidence >= 0.65 creates track, persists to storage, schedules alarm & broadcasts
    await doInstance.webSocketMessage(
      ws as unknown as WebSocket,
      JSON.stringify({
        nodeId: "node-a",
        timestamp: now + 500,
        location: { lat: 50.45, lng: 30.52, accuracy: 10 },
        detection: {
          class: "Shahed_Piston_UAV",
          confidence: 0.89,
          snr_db: 18,
          peak_freq_hz: 92.5
        }
      })
    );
    expect(ws.sent).toHaveLength(3);
    const detBroadcast = JSON.parse(ws.sent[2]) as TrackUpdateMessage;
    expect(detBroadcast.tracks).toHaveLength(1);
    expect(detBroadcast.tracks[0].class).toBe("Shahed_Piston_UAV");
    expect(state.storage.alarmAt).not.toBeNull();

    // Simulate Durable Object hibernation wake-up: construct fresh DO instance with same storage
    const hibernatedDO = new SpatialZoneDO(
      state as unknown as DurableObjectState,
      TEST_ENV
    );
    const stateRes = await hibernatedDO.fetch(
      new Request("https://mesh.local/state", {
        headers: { "x-cell-id": cellId }
      })
    );
    expect(stateRes.status).toBe(200);
    const restoredSnapshot = (await stateRes.json()) as SnapshotMessage;
    expect(restoredSnapshot.cellId).toBe(cellId);
    expect(restoredSnapshot.tracks).toHaveLength(1);
    expect(restoredSnapshot.nodes).toHaveLength(1);
  });

  it("handles POST /ingest, GET /state, alarm cleanup, and WebSocket close", async () => {
    vi.useFakeTimers();
    const baseTime = 1_750_000_000_000;
    vi.setSystemTime(baseTime);

    const state = new MockDurableObjectState();
    const doInstance = new SpatialZoneDO(
      state as unknown as DurableObjectState,
      TEST_ENV
    );

    await doInstance.fetch(
      new Request("https://mesh.local/ws", {
        headers: {
          Upgrade: "websocket",
          "x-cell-id": cellId,
          "x-node-id": "node-ws"
        }
      })
    );
    const ws = state.acceptedSockets[0];

    // Sub-threshold POST /ingest -> 202 accepted: false
    const lowConfRes = await doInstance.fetch(
      new Request("https://mesh.local/ingest", {
        method: "POST",
        headers: { "content-type": "application/json", "x-cell-id": cellId },
        body: JSON.stringify({
          nodeId: "node-http",
          timestamp: baseTime,
          location: { lat: 50.451, lng: 30.521, accuracy: 12 },
          detection: {
            class: "Multirotor_UAV",
            confidence: 0.5,
            snr_db: 10,
            peak_freq_hz: 180
          }
        })
      })
    );
    expect(lowConfRes.status).toBe(202);
    const lowBody = (await lowConfRes.json()) as { accepted: boolean; tracks: unknown[] };
    expect(lowBody.accepted).toBe(false);
    expect(lowBody.tracks).toHaveLength(0);

    // Accepted POST /ingest -> 200 accepted: true & broadcasts
    const okRes = await doInstance.fetch(
      new Request("https://mesh.local/ingest", {
        method: "POST",
        headers: { "content-type": "application/json", "x-cell-id": cellId },
        body: JSON.stringify({
          nodeId: "node-http",
          timestamp: baseTime,
          location: { lat: 50.451, lng: 30.521, accuracy: 12 },
          detection: {
            class: "Multirotor_UAV",
            confidence: 0.85,
            snr_db: 16,
            peak_freq_hz: 180
          }
        })
      })
    );
    expect(okRes.status).toBe(200);
    const okBody = (await okRes.json()) as { accepted: boolean; tracks: unknown[] };
    expect(okBody.accepted).toBe(true);
    expect(okBody.tracks).toHaveLength(1);
    expect(ws.sent).toHaveLength(2);

    // Advance 16 seconds: track becomes stale (> 15000 ms), node-http still active (< 45000 ms)
    vi.setSystemTime(baseTime + 16_000);
    await doInstance.alarm();
    expect(ws.sent).toHaveLength(3);
    const afterTrackPrune = JSON.parse(ws.sent[2]) as TrackUpdateMessage;
    expect(afterTrackPrune.tracks).toHaveLength(0);
    expect(afterTrackPrune.nodes).toHaveLength(1);

    // Advance past 45 seconds: stale node pruned too
    vi.setSystemTime(baseTime + 50_000);
    await doInstance.alarm();
    const afterNodePrune = JSON.parse(ws.sent[3]) as TrackUpdateMessage;
    expect(afterNodePrune.nodes).toHaveLength(0);

    // Close WebSocket cleanly
    await doInstance.webSocketClose(ws as unknown as WebSocket, 1000, "Normal", true);
    expect(ws.closed).toBe(true);
    expect(state.getWebSockets()).toHaveLength(0);
  });
});
