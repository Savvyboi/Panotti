import { cellToBoundary, gridDisk, isValidCell, latLngToCell } from "h3-js";
import { SpatialZoneDO } from "./SpatialZoneDO";
import { parseClientMessage, type Env } from "./types";

export { SpatialZoneDO };

const H3_RESOLUTION = 6;

/**
 * Checks whether an Origin header is permitted by env.ALLOWED_ORIGINS.
 * When ALLOWED_ORIGINS includes any http://localhost or http://127.0.0.1 entry,
 * any port on http://localhost:* or http://127.0.0.1:* is also allowed for dev.
 */
export function isOriginAllowed(
  originHeader: string | null,
  allowedOriginsRaw?: string
): boolean {
  if (!originHeader || !allowedOriginsRaw) {
    return false;
  }

  let parsedOrigin: URL;
  try {
    parsedOrigin = new URL(originHeader);
  } catch {
    return false;
  }

  if (parsedOrigin.origin !== originHeader) {
    return false;
  }
  if (parsedOrigin.protocol !== "http:" && parsedOrigin.protocol !== "https:") {
    return false;
  }

  const allowedEntries = allowedOriginsRaw
    .split(",")
    .map((item) => item.trim())
    .filter((item) => item.length > 0);

  if (allowedEntries.includes(originHeader)) {
    return true;
  }

  const hasLocalhostDevEntry = allowedEntries.some((entry) => {
    try {
      const u = new URL(entry);
      return (
        u.protocol === "http:" &&
        (u.hostname === "localhost" || u.hostname === "127.0.0.1")
      );
    } catch {
      return false;
    }
  });

  if (
    hasLocalhostDevEntry &&
    parsedOrigin.protocol === "http:" &&
    (parsedOrigin.hostname === "localhost" || parsedOrigin.hostname === "127.0.0.1")
  ) {
    return true;
  }

  return false;
}

function buildCorsHeaders(
  originHeader: string | null,
  allowedOriginsRaw?: string
): Headers {
  const headers = new Headers();
  headers.set("Vary", "Origin");
  if (originHeader && isOriginAllowed(originHeader, allowedOriginsRaw)) {
    headers.set("Access-Control-Allow-Origin", originHeader);
    headers.set("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    headers.set("Access-Control-Allow-Headers", "Content-Type");
    headers.set("Access-Control-Max-Age", "86400");
  }
  return headers;
}

function withCors(
  response: Response,
  originHeader: string | null,
  allowedOriginsRaw?: string
): Response {
  const headers = new Headers(response.headers);
  const cors = buildCorsHeaders(originHeader, allowedOriginsRaw);
  cors.forEach((value, key) => {
    headers.set(key, value);
  });
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers
  });
}

function jsonResponse(
  data: unknown,
  status: number,
  originHeader: string | null,
  allowedOriginsRaw?: string
): Response {
  const headers = buildCorsHeaders(originHeader, allowedOriginsRaw);
  headers.set("content-type", "application/json; charset=utf-8");
  return new Response(JSON.stringify(data), { status, headers });
}

function parseCoordinatePair(
  latRaw: string | null,
  lngRaw: string | null
): { ok: true; lat: number; lng: number } | { ok: false; error: string } {
  if (latRaw === null || lngRaw === null || latRaw.trim() === "" || lngRaw.trim() === "") {
    return { ok: false, error: "lat and lng query parameters are required" };
  }
  const lat = Number(latRaw);
  const lng = Number(lngRaw);
  if (!Number.isFinite(lat) || lat < -90 || lat > 90) {
    return { ok: false, error: "lat must be a finite number in [-90, 90]" };
  }
  if (!Number.isFinite(lng) || lng < -180 || lng > 180) {
    return { ok: false, error: "lng must be a finite number in [-180, 180]" };
  }
  return { ok: true, lat, lng };
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    const originHeader = request.headers.get("Origin");

    if (request.method === "OPTIONS") {
      if (originHeader && !isOriginAllowed(originHeader, env.ALLOWED_ORIGINS)) {
        return new Response(null, { status: 403 });
      }
      return new Response(null, {
        status: 204,
        headers: buildCorsHeaders(originHeader, env.ALLOWED_ORIGINS)
      });
    }

    if (request.method === "GET" && url.pathname === "/health") {
      return jsonResponse(
        { status: "ok", service: "panotti-mesh-hub" },
        200,
        originHeader,
        env.ALLOWED_ORIGINS
      );
    }

    if (request.method === "GET" && url.pathname === "/api/cell") {
      const coords = parseCoordinatePair(
        url.searchParams.get("lat"),
        url.searchParams.get("lng")
      );
      if (!coords.ok) {
        return jsonResponse(
          { error: coords.error },
          400,
          originHeader,
          env.ALLOWED_ORIGINS
        );
      }

      const cellId = latLngToCell(coords.lat, coords.lng, H3_RESOLUTION);
      const boundary = cellToBoundary(cellId);
      const disk = gridDisk(cellId, 1);

      return jsonResponse(
        {
          cellId,
          resolution: H3_RESOLUTION,
          boundary,
          gridDisk: disk
        },
        200,
        originHeader,
        env.ALLOWED_ORIGINS
      );
    }

    if (request.method === "GET" && url.pathname === "/ws") {
      if (!isOriginAllowed(originHeader, env.ALLOWED_ORIGINS)) {
        return jsonResponse(
          { error: "Forbidden origin" },
          403,
          originHeader,
          env.ALLOWED_ORIGINS
        );
      }

      if (request.headers.get("Upgrade")?.toLowerCase() !== "websocket") {
        return jsonResponse(
          { error: "Expected Upgrade: websocket" },
          426,
          originHeader,
          env.ALLOWED_ORIGINS
        );
      }

      const cellParam = url.searchParams.get("cell");
      let cellId: string | null = null;
      if (cellParam !== null && cellParam.trim() !== "") {
        if (!isValidCell(cellParam)) {
          return jsonResponse(
            { error: "Invalid H3 cell parameter" },
            400,
            originHeader,
            env.ALLOWED_ORIGINS
          );
        }
        cellId = cellParam;
      } else {
        const coords = parseCoordinatePair(
          url.searchParams.get("lat"),
          url.searchParams.get("lng")
        );
        if (!coords.ok) {
          return jsonResponse(
            { error: coords.error },
            400,
            originHeader,
            env.ALLOWED_ORIGINS
          );
        }
        cellId = latLngToCell(coords.lat, coords.lng, H3_RESOLUTION);
      }

      const rawNodeId = url.searchParams.get("nodeId") ?? "observer";
      const nodeId = rawNodeId.trim();
      if (nodeId.length === 0 || nodeId.length > 128) {
        return jsonResponse(
          { error: "nodeId must be 1..128 characters" },
          400,
          originHeader,
          env.ALLOWED_ORIGINS
        );
      }

      const id = env.SPATIAL_ZONE.idFromName(cellId);
      const stub = env.SPATIAL_ZONE.get(id);
      const headers = new Headers(request.headers);
      headers.set("x-cell-id", cellId);
      headers.set("x-node-id", nodeId);

      const doRequest = new Request(request.url, {
        method: request.method,
        headers
      });
      return stub.fetch(doRequest);
    }

    if (request.method === "POST" && url.pathname === "/api/telemetry") {
      let body: unknown;
      try {
        body = await request.json();
      } catch {
        return jsonResponse(
          { error: "Invalid JSON payload" },
          400,
          originHeader,
          env.ALLOWED_ORIGINS
        );
      }

      const parsed = parseClientMessage(body);
      if (!parsed.ok) {
        return jsonResponse(
          { error: parsed.error },
          400,
          originHeader,
          env.ALLOWED_ORIGINS
        );
      }

      const cellId = latLngToCell(
        parsed.message.location.lat,
        parsed.message.location.lng,
        H3_RESOLUTION
      );
      const id = env.SPATIAL_ZONE.idFromName(cellId);
      const stub = env.SPATIAL_ZONE.get(id);

      const doUrl = new URL(request.url);
      doUrl.pathname = "/ingest";
      const doResponse = await stub.fetch(
        new Request(doUrl.toString(), {
          method: "POST",
          headers: {
            "content-type": "application/json; charset=utf-8",
            "x-cell-id": cellId
          },
          body: JSON.stringify(parsed.message)
        })
      );

      return withCors(doResponse, originHeader, env.ALLOWED_ORIGINS);
    }

    const stateMatch = url.pathname.match(/^\/api\/zone\/([^/]+)\/state$/);
    if (request.method === "GET" && stateMatch) {
      const cellId = decodeURIComponent(stateMatch[1]);
      if (!isValidCell(cellId)) {
        return jsonResponse(
          { error: "Invalid H3 cellId" },
          400,
          originHeader,
          env.ALLOWED_ORIGINS
        );
      }

      const id = env.SPATIAL_ZONE.idFromName(cellId);
      const stub = env.SPATIAL_ZONE.get(id);

      const doUrl = new URL(request.url);
      doUrl.pathname = "/state";
      const doResponse = await stub.fetch(
        new Request(doUrl.toString(), {
          method: "GET",
          headers: {
            "x-cell-id": cellId
          }
        })
      );

      return withCors(doResponse, originHeader, env.ALLOWED_ORIGINS);
    }

    return jsonResponse(
      { error: "Not found" },
      404,
      originHeader,
      env.ALLOWED_ORIGINS
    );
  }
};
