import { latLngToCell } from "h3-js";
import { describe, expect, it, vi } from "vitest";
import worker, { isOriginAllowed } from "../src/index";
import { parseClientMessage, type Env } from "../src/types";

const DEFAULT_ALLOWED_ORIGINS =
  "http://localhost:4173,http://localhost:5173,http://127.0.0.1:4173,http://127.0.0.1:5173";

function createMockEnv(
  stubHandler: (request: Request, cellName: string) => Promise<Response> | Response = () =>
    new Response(JSON.stringify({ ok: true }), {
      status: 200,
      headers: { "content-type": "application/json" }
    })
): {
  env: Env;
  calls: Array<{ cellName: string; request: Request }>;
} {
  const calls: Array<{ cellName: string; request: Request }> = [];
  const env: Env = {
    ALLOWED_ORIGINS: DEFAULT_ALLOWED_ORIGINS,
    SPATIAL_ZONE: {
      idFromName(name: string) {
        return { toString: () => name, name } as unknown as DurableObjectId;
      },
      get(id: DurableObjectId) {
        const cellName = (id as unknown as { name?: string }).name ?? id.toString();
        return {
          fetch: vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
            const req = input instanceof Request ? input : new Request(input, init);
            calls.push({ cellName, request: req });
            return stubHandler(req, cellName);
          })
        } as unknown as DurableObjectStub;
      }
    } as unknown as DurableObjectNamespace
  };
  return { env, calls };
}

describe("parseClientMessage validation", () => {
  it("accepts raw detection payload without type and typed detection payload", () => {
    const rawNoType = {
      nodeId: "node-alpha",
      timestamp: 1700000000000,
      location: { lat: 50.45, lng: 30.52, accuracy: 8 },
      detection: {
        class: "Shahed_Piston_UAV",
        confidence: 0.87,
        snr_db: 14.5,
        peak_freq_hz: 92.5
      }
    };
    const res1 = parseClientMessage(rawNoType);
    expect(res1.ok).toBe(true);
    if (res1.ok) {
      expect(res1.message.type).toBe("detection");
    }

    const res2 = parseClientMessage(
      JSON.stringify({ type: "detection", ...rawNoType })
    );
    expect(res2.ok).toBe(true);
  });

  it("rejects out-of-bounds lat/lng/confidence/snr and invalid classes", () => {
    const base = {
      nodeId: "node-1",
      timestamp: 1700000000000,
      location: { lat: 50, lng: 30, accuracy: 5 },
      detection: {
        class: "Multirotor_UAV",
        confidence: 0.8,
        snr_db: 12,
        peak_freq_hz: 240
      }
    };

    expect(
      parseClientMessage({ ...base, location: { lat: 91, lng: 30, accuracy: 5 } }).ok
    ).toBe(false);
    expect(
      parseClientMessage({ ...base, location: { lat: 50, lng: -181, accuracy: 5 } }).ok
    ).toBe(false);
    expect(
      parseClientMessage({
        ...base,
        detection: { ...base.detection, confidence: 1.05 }
      }).ok
    ).toBe(false);
    expect(
      parseClientMessage({
        ...base,
        detection: { ...base.detection, confidence: -0.1 }
      }).ok
    ).toBe(false);
    expect(
      parseClientMessage({
        ...base,
        detection: { ...base.detection, snr_db: 250 }
      }).ok
    ).toBe(false);
    expect(
      parseClientMessage({
        ...base,
        detection: { ...base.detection, class: "Unknown_Plane" }
      }).ok
    ).toBe(false);
    expect(parseClientMessage("{bad json").ok).toBe(false);
  });
});

describe("WorkerRouter", () => {
  it("handles GET /health", async () => {
    const { env } = createMockEnv();
    const res = await worker.fetch(new Request("https://mesh.local/health"), env);
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({
      status: "ok",
      service: "panotti-mesh-hub"
    });
  });

  it("computes H3 Res 6 cell, boundary, and gridDisk on GET /api/cell", async () => {
    const { env } = createMockEnv();
    const res = await worker.fetch(
      new Request("https://mesh.local/api/cell?lat=50.45&lng=30.52", {
        headers: { Origin: "http://localhost:5173" }
      }),
      env
    );
    expect(res.status).toBe(200);
    expect(res.headers.get("Access-Control-Allow-Origin")).toBe(
      "http://localhost:5173"
    );

    const data = (await res.json()) as {
      cellId: string;
      resolution: number;
      boundary: Array<[number, number]>;
      gridDisk: string[];
    };
    expect(data.cellId).toBe(latLngToCell(50.45, 30.52, 6));
    expect(data.resolution).toBe(6);
    expect(data.boundary).toHaveLength(6);
    expect(data.gridDisk).toHaveLength(7);
    expect(data.gridDisk[0]).toBe(data.cellId);

    const bad = await worker.fetch(
      new Request("https://mesh.local/api/cell?lat=99&lng=30"),
      env
    );
    expect(bad.status).toBe(400);
  });

  it("validates Origin strictly on GET /ws and forwards allowed upgrades to SpatialZoneDO", async () => {
    const { env, calls } = createMockEnv();

    // Missing Origin -> 403
    const missingOrigin = await worker.fetch(
      new Request("https://mesh.local/ws?lat=50.45&lng=30.52", {
        headers: { Upgrade: "websocket" }
      }),
      env
    );
    expect(missingOrigin.status).toBe(403);

    // Untrusted Origin -> 403
    const badOrigin = await worker.fetch(
      new Request("https://mesh.local/ws?lat=50.45&lng=30.52", {
        headers: {
          Origin: "https://evil.example",
          Upgrade: "websocket"
        }
      }),
      env
    );
    expect(badOrigin.status).toBe(403);

    // Subdomain spoofing -> 403
    expect(
      isOriginAllowed("http://localhost.evil.com", DEFAULT_ALLOWED_ORIGINS)
    ).toBe(false);

    // Missing Upgrade header -> 426
    const noUpgrade = await worker.fetch(
      new Request("https://mesh.local/ws?lat=50.45&lng=30.52", {
        headers: { Origin: "http://localhost:5173" }
      }),
      env
    );
    expect(noUpgrade.status).toBe(426);

    // Allowed localhost dev port -> forwards to DO with x-cell-id and x-node-id
    const expectedCell = latLngToCell(50.45, 30.52, 6);
    const okRes = await worker.fetch(
      new Request("https://mesh.local/ws?lat=50.45&lng=30.52&nodeId=sensor-9", {
        headers: {
          Origin: "http://localhost:9999",
          Upgrade: "websocket"
        }
      }),
      env
    );
    expect(okRes.status).toBe(200);
    expect(calls).toHaveLength(1);
    expect(calls[0].cellName).toBe(expectedCell);
    expect(calls[0].request.headers.get("x-cell-id")).toBe(expectedCell);
    expect(calls[0].request.headers.get("x-node-id")).toBe("sensor-9");

    // Valid ?cell= param also routes cleanly, invalid ?cell= returns 400
    const badCell = await worker.fetch(
      new Request("https://mesh.local/ws?cell=not-a-cell", {
        headers: {
          Origin: "http://127.0.0.1:4173",
          Upgrade: "websocket"
        }
      }),
      env
    );
    expect(badCell.status).toBe(400);

    const goodCell = await worker.fetch(
      new Request(`https://mesh.local/ws?cell=${expectedCell}&nodeId=sensor-10`, {
        headers: {
          Origin: "http://127.0.0.1:8080",
          Upgrade: "websocket"
        }
      }),
      env
    );
    expect(goodCell.status).toBe(200);
    expect(calls).toHaveLength(2);
    expect(calls[1].cellName).toBe(expectedCell);
  });

  it("validates and routes POST /api/telemetry and GET /api/zone/:cellId/state with CORS", async () => {
    const { env, calls } = createMockEnv();
    const expectedCell = latLngToCell(50.45, 30.52, 6);

    // OPTIONS preflight
    const preflightOk = await worker.fetch(
      new Request("https://mesh.local/api/telemetry", {
        method: "OPTIONS",
        headers: { Origin: "http://localhost:5173" }
      }),
      env
    );
    expect(preflightOk.status).toBe(204);
    expect(preflightOk.headers.get("Access-Control-Allow-Origin")).toBe(
      "http://localhost:5173"
    );

    const preflightBad = await worker.fetch(
      new Request("https://mesh.local/api/telemetry", {
        method: "OPTIONS",
        headers: { Origin: "https://attacker.example" }
      }),
      env
    );
    expect(preflightBad.status).toBe(403);

    // Invalid telemetry -> 400
    const badTelemetry = await worker.fetch(
      new Request("https://mesh.local/api/telemetry", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ nodeId: "n1", timestamp: -1 })
      }),
      env
    );
    expect(badTelemetry.status).toBe(400);

    // Valid telemetry -> forwards to DO /ingest
    const okTelemetry = await worker.fetch(
      new Request("https://mesh.local/api/telemetry", {
        method: "POST",
        headers: {
          "content-type": "application/json",
          Origin: "http://localhost:4173"
        },
        body: JSON.stringify({
          nodeId: "n1",
          timestamp: 1700000000000,
          location: { lat: 50.45, lng: 30.52, accuracy: 12 },
          detection: {
            class: "Shahed_Piston_UAV",
            confidence: 0.92,
            snr_db: 18,
            peak_freq_hz: 92.5
          }
        })
      }),
      env
    );
    expect(okTelemetry.status).toBe(200);
    expect(okTelemetry.headers.get("Access-Control-Allow-Origin")).toBe(
      "http://localhost:4173"
    );
    expect(calls[0].cellName).toBe(expectedCell);
    expect(new URL(calls[0].request.url).pathname).toBe("/ingest");

    // GET /api/zone/:cellId/state
    const badZone = await worker.fetch(
      new Request("https://mesh.local/api/zone/invalid_hex/state"),
      env
    );
    expect(badZone.status).toBe(400);

    const okZone = await worker.fetch(
      new Request(`https://mesh.local/api/zone/${expectedCell}/state`, {
        headers: { Origin: "http://localhost:5173" }
      }),
      env
    );
    expect(okZone.status).toBe(200);
    expect(okZone.headers.get("Access-Control-Allow-Origin")).toBe(
      "http://localhost:5173"
    );
    expect(calls[1].cellName).toBe(expectedCell);
    expect(new URL(calls[1].request.url).pathname).toBe("/state");
  });
});
