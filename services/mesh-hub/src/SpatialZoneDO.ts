import { DurableObject } from "cloudflare:workers";
import { cellToBoundary, cellToLatLng, isValidCell } from "h3-js";
import { BearingFusion, type SerializedBearingFusion } from "./tracker/BearingFusion";
import {
  parseClientMessage,
  type Env,
  type NodeState,
  type SnapshotMessage,
  type TrackUpdateMessage
} from "./types";

export const MIN_DETECTION_CONFIDENCE = 0.65;
export const NODE_STALE_MS = 45_000;
export const CLEANUP_ALARM_INTERVAL_MS = 5_000;

const STORAGE_KEY_CELL_ID = "cellId";
const STORAGE_KEY_NODES = "nodes";
const STORAGE_KEY_FUSION = "fusion";

interface SocketAttachment {
  nodeId: string;
  cellId: string;
}

/**
 * SpatialZoneDO manages real-time acoustic telemetry, BearingFusion/Kalman tracking,
 * node heartbeats, and WebSocket Hibernation fan-out for a single H3 Resolution-6 hexagon.
 */
export class SpatialZoneDO extends DurableObject<Env> {
  private cellId: string | null = null;
  private nodes = new Map<string, NodeState>();
  private fusion = new BearingFusion();
  private loaded = false;
  private readonly loadPromise: Promise<void>;

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    this.loadPromise = this.ctx.blockConcurrencyWhile(async () => {
      await this.restoreFromStorage();
    });
  }

  private async ensureLoaded(): Promise<void> {
    await this.loadPromise;
    if (!this.loaded) {
      await this.restoreFromStorage();
    }
  }

  private async restoreFromStorage(): Promise<void> {
    const [storedCellId, storedNodes, storedFusion] = await Promise.all([
      this.ctx.storage.get<string>(STORAGE_KEY_CELL_ID),
      this.ctx.storage.get<NodeState[]>(STORAGE_KEY_NODES),
      this.ctx.storage.get<SerializedBearingFusion>(STORAGE_KEY_FUSION)
    ]);

    if (typeof storedCellId === "string" && isValidCell(storedCellId)) {
      this.cellId = storedCellId;
    }

    this.nodes.clear();
    if (Array.isArray(storedNodes)) {
      for (const node of storedNodes) {
        if (node && typeof node.nodeId === "string") {
          this.nodes.set(node.nodeId, node);
        }
      }
    }

    if (storedFusion) {
      this.fusion = BearingFusion.restore(storedFusion);
    } else if (this.cellId) {
      const [originLat, originLng] = cellToLatLng(this.cellId);
      this.fusion = new BearingFusion({ originLat, originLng });
    }

    this.loaded = true;
  }

  private async persistState(): Promise<void> {
    const puts: Record<string, unknown> = {
      [STORAGE_KEY_NODES]: Array.from(this.nodes.values()),
      [STORAGE_KEY_FUSION]: this.fusion.serialize()
    };
    if (this.cellId !== null) {
      puts[STORAGE_KEY_CELL_ID] = this.cellId;
    }
    await this.ctx.storage.put(puts);
  }

  private async scheduleCleanupAlarm(): Promise<void> {
    const hasActiveState =
      this.fusion.getTracks().length > 0 ||
      this.fusion.getBufferedDetectionCount() > 0 ||
      this.nodes.size > 0;
    if (!hasActiveState) {
      return;
    }
    const existingAlarm = await this.ctx.storage.getAlarm();
    if (existingAlarm === null) {
      await this.ctx.storage.setAlarm(Date.now() + CLEANUP_ALARM_INTERVAL_MS);
    }
  }

  private ensureCellInitialized(cellId: string | null): void {
    if (!cellId || !isValidCell(cellId)) {
      return;
    }
    if (this.cellId !== cellId) {
      this.cellId = cellId;
      const [originLat, originLng] = cellToLatLng(cellId);
      if (
        this.fusion.getTracks().length === 0 &&
        this.fusion.getBufferedDetectionCount() === 0
      ) {
        this.fusion = new BearingFusion({ originLat, originLng });
      }
    }
  }

  private getBoundary(): Array<[number, number]> {
    if (!this.cellId || !isValidCell(this.cellId)) {
      return [];
    }
    return cellToBoundary(this.cellId);
  }

  private buildSnapshot(): SnapshotMessage {
    return {
      type: "snapshot",
      cellId: this.cellId ?? "",
      boundary: this.getBoundary(),
      tracks: this.fusion.getTracks(),
      nodes: Array.from(this.nodes.values())
    };
  }

  private buildTrackUpdate(timestamp = Date.now()): TrackUpdateMessage {
    return {
      type: "track_update",
      cellId: this.cellId ?? "",
      timestamp,
      boundary: this.getBoundary(),
      tracks: this.fusion.getTracks(),
      nodes: Array.from(this.nodes.values())
    };
  }

  private broadcastTrackUpdate(timestamp = Date.now()): void {
    const sockets = this.ctx.getWebSockets();
    if (sockets.length === 0) {
      return;
    }
    const payload = JSON.stringify(this.buildTrackUpdate(timestamp));
    for (const ws of sockets) {
      try {
        ws.send(payload);
      } catch {
        // Ignore broken sockets; webSocketClose/webSocketError handles cleanup.
      }
    }
  }

  private resolveCellIdFromRequest(request: Request, url: URL): string | null {
    const headerCell = request.headers.get("x-cell-id");
    if (headerCell && isValidCell(headerCell)) {
      return headerCell;
    }
    const queryCell = url.searchParams.get("cellId") ?? url.searchParams.get("cell");
    if (queryCell && isValidCell(queryCell)) {
      return queryCell;
    }
    return this.cellId;
  }

  override async fetch(request: Request): Promise<Response> {
    await this.ensureLoaded();

    const url = new URL(request.url);
    const resolvedCellId = this.resolveCellIdFromRequest(request, url);
    if (resolvedCellId) {
      this.ensureCellInitialized(resolvedCellId);
    }

    if (request.headers.get("Upgrade")?.toLowerCase() === "websocket") {
      if (!this.cellId) {
        return jsonResponse({ error: "Missing or invalid H3 cellId" }, 400);
      }

      const rawNodeId =
        request.headers.get("x-node-id") ??
        url.searchParams.get("nodeId") ??
        "observer";
      const nodeId = rawNodeId.trim().slice(0, 128) || "observer";

      const pair = new WebSocketPair();
      const client = pair[0];
      const server = pair[1];

      this.ctx.acceptWebSocket(server);
      const attachment: SocketAttachment = { nodeId, cellId: this.cellId };
      server.serializeAttachment(attachment);

      // Persist cellId if this is the first request establishing this DO's zone
      await this.persistState();

      server.send(JSON.stringify(this.buildSnapshot()));
      return new Response(null, { status: 101, webSocket: client });
    }

    if (request.method === "GET" && url.pathname.endsWith("/state")) {
      return jsonResponse(this.buildSnapshot(), 200);
    }

    if (request.method === "POST" && url.pathname.endsWith("/ingest")) {
      let body: unknown;
      try {
        body = await request.json();
      } catch {
        return jsonResponse({ error: "Invalid JSON body" }, 400);
      }

      const result = await this.handleClientMessage(body);
      if (!result.ok) {
        return jsonResponse({ error: result.error }, 400);
      }
      if (!result.accepted) {
        return jsonResponse(
          {
            ok: true,
            accepted: false,
            reason: result.reason,
            cellId: this.cellId ?? "",
            tracks: this.fusion.getTracks(),
            nodes: Array.from(this.nodes.values())
          },
          202
        );
      }
      return jsonResponse(
        {
          ok: true,
          accepted: true,
          cellId: this.cellId ?? "",
          tracks: this.fusion.getTracks(),
          nodes: Array.from(this.nodes.values())
        },
        200
      );
    }

    return jsonResponse({ error: "Not found" }, 404);
  }

  override async webSocketMessage(
    ws: WebSocket,
    message: string | ArrayBuffer
  ): Promise<void> {
    await this.ensureLoaded();

    const attachment = ws.deserializeAttachment() as SocketAttachment | null;
    if (attachment?.cellId && !this.cellId) {
      this.ensureCellInitialized(attachment.cellId);
    }

    let rawText: string;
    if (typeof message === "string") {
      rawText = message;
    } else {
      rawText = new TextDecoder().decode(message);
    }

    const result = await this.handleClientMessage(rawText);
    if (!result.ok) {
      try {
        ws.send(JSON.stringify({ type: "error", error: result.error }));
      } catch {
        // Ignore send errors on closing socket
      }
    }
  }

  override async webSocketClose(
    ws: WebSocket,
    code: number,
    reason: string,
    _wasClean: boolean
  ): Promise<void> {
    await this.ensureLoaded();
    await this.cleanupSocket(ws, code, reason);
  }

  override async webSocketError(ws: WebSocket, _error: unknown): Promise<void> {
    await this.ensureLoaded();
    await this.cleanupSocket(ws, 1011, "WebSocket error");
  }

  override async alarm(): Promise<void> {
    await this.ensureLoaded();

    const now = Date.now();
    const tracksPruned = this.fusion.prune(now);

    let nodesPruned = false;
    for (const [nodeId, node] of this.nodes.entries()) {
      if (now - node.lastSeen > NODE_STALE_MS) {
        this.nodes.delete(nodeId);
        nodesPruned = true;
      }
    }

    if (tracksPruned || nodesPruned) {
      await this.persistState();
      this.broadcastTrackUpdate(now);
    }

    const hasActiveState =
      this.fusion.getTracks().length > 0 ||
      this.fusion.getBufferedDetectionCount() > 0 ||
      this.nodes.size > 0;
    if (hasActiveState) {
      await this.ctx.storage.setAlarm(Date.now() + CLEANUP_ALARM_INTERVAL_MS);
    }
  }

  private async handleClientMessage(
    raw: unknown
  ): Promise<
    | { ok: true; accepted: boolean; reason?: string }
    | { ok: false; error: string }
  > {
    const parsed = parseClientMessage(raw);
    if (!parsed.ok) {
      return parsed;
    }

    const msg = parsed.message;
    if (msg.type === "heartbeat") {
      this.nodes.set(msg.nodeId, {
        nodeId: msg.nodeId,
        lat: msg.location.lat,
        lng: msg.location.lng,
        ...(msg.location.accuracy !== undefined
          ? { accuracy: msg.location.accuracy }
          : {}),
        lastSeen: msg.timestamp
      });

      await this.persistState();
      await this.scheduleCleanupAlarm();
      this.broadcastTrackUpdate(msg.timestamp);
      return { ok: true, accepted: true };
    }

    // Enforce confidence >= 0.65 gate for incoming detections
    if (msg.detection.confidence < MIN_DETECTION_CONFIDENCE) {
      return {
        ok: true,
        accepted: false,
        reason: `Detection confidence ${msg.detection.confidence} is below ${MIN_DETECTION_CONFIDENCE}`
      };
    }

    this.nodes.set(msg.nodeId, {
      nodeId: msg.nodeId,
      lat: msg.location.lat,
      lng: msg.location.lng,
      accuracy: msg.location.accuracy,
      lastSeen: msg.timestamp
    });

    this.fusion.ingest(msg);
    await this.persistState();
    await this.scheduleCleanupAlarm();
    this.broadcastTrackUpdate(msg.timestamp);

    return { ok: true, accepted: true };
  }

  private async cleanupSocket(
    ws: WebSocket,
    code: number,
    reason: string
  ): Promise<void> {
    const attachment = ws.deserializeAttachment() as SocketAttachment | null;
    try {
      ws.close(code, reason);
    } catch {
      // Socket may already be closed
    }

    if (!attachment?.nodeId) {
      return;
    }

    const remainingForNode = this.ctx.getWebSockets().some((other) => {
      if (other === ws) {
        return false;
      }
      const otherMeta = other.deserializeAttachment() as SocketAttachment | null;
      return otherMeta?.nodeId === attachment.nodeId;
    });

    if (!remainingForNode && this.nodes.delete(attachment.nodeId)) {
      await this.persistState();
      this.broadcastTrackUpdate();
    }
  }
}

function jsonResponse(data: unknown, status: number): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "content-type": "application/json; charset=utf-8" }
  });
}
