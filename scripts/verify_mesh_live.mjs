/**
 * Live end-to-end verification script against a running Cloudflare Wrangler mesh-hub
 * (e.g. `npx.cmd wrangler dev --port 8787` on http://127.0.0.1:8787).
 */

import assert from 'node:assert/strict';
import http from 'node:http';

const BASE_HTTP = process.env.PANOTTI_MESH_HTTP || 'http://127.0.0.1:8787';
const BASE_WS = BASE_HTTP.replace(/^http/i, 'ws');
const TRUSTED_ORIGIN = 'http://localhost:4173';

const NODE_ALPHA = {
  nodeId: 'e2e-alpha',
  lat: 50.4501,
  lng: 30.5234,
  accuracy: 4.0,
};

const NODE_BETA = {
  nodeId: 'e2e-beta',
  lat: 50.4501,
  lng: 30.532,
  accuracy: 6.0,
};

function openWsWithOrigin(url, origin) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(url, { headers: { Origin: origin } });
    const messages = [];
    const waiters = [];

    const timer = setTimeout(() => {
      reject(new Error(`WebSocket connect timeout for ${url}`));
    }, 6000);

    ws.addEventListener('message', (event) => {
      const parsed = JSON.parse(String(event.data));
      messages.push(parsed);
      for (let i = waiters.length - 1; i >= 0; i -= 1) {
        if (waiters[i].predicate(parsed)) {
          const w = waiters.splice(i, 1)[0];
          clearTimeout(w.timeoutId);
          w.resolve(parsed);
        }
      }
    });

    ws.addEventListener('open', () => {
      clearTimeout(timer);
      resolve({
        ws,
        messages,
        waitFor(predicate, timeoutMs = 6000) {
          const existing = messages.find(predicate);
          if (existing) return Promise.resolve(existing);
          return new Promise((res, rej) => {
            const timeoutId = setTimeout(() => {
              rej(new Error(`Timed out waiting for WebSocket message (${timeoutMs} ms)`));
            }, timeoutMs);
            waiters.push({ predicate, resolve: res, timeoutId });
          });
        },
      });
    });

    ws.addEventListener('error', (err) => {
      clearTimeout(timer);
      reject(new Error(`WebSocket error for ${url}: ${err?.message ?? 'unknown'}`));
    });
  });
}

async function main() {
  // 1. Health check
  const healthRes = await fetch(`${BASE_HTTP}/health`);
  assert.equal(healthRes.status, 200);
  const health = await healthRes.json();
  assert.equal(health.status, 'ok');
  assert.equal(health.service, 'panotti-mesh-hub');

  // 2. H3 Res 6 cell resolution check
  const cellRes = await fetch(
    `${BASE_HTTP}/api/cell?lat=${NODE_ALPHA.lat}&lng=${NODE_ALPHA.lng}`
  );
  assert.equal(cellRes.status, 200);
  const cellData = await cellRes.json();
  assert.equal(typeof cellData.cellId, 'string');
  assert.equal(cellData.resolution, 6);
  assert.equal(cellData.boundary.length, 6);
  assert.equal(cellData.gridDisk.length, 7);

  // 3. Strict Origin validation on /ws handshake (real Upgrade: websocket request)
  const badOriginStatus = await new Promise((resolve, reject) => {
    const target = new URL(
      `${BASE_HTTP}/ws?lat=${NODE_ALPHA.lat}&lng=${NODE_ALPHA.lng}&nodeId=untrusted-probe`
    );
    const req = http.request(
      {
        hostname: target.hostname,
        port: target.port,
        path: `${target.pathname}${target.search}`,
        method: 'GET',
        headers: {
          Origin: 'https://untrusted.example',
          Connection: 'Upgrade',
          Upgrade: 'websocket',
          'Sec-WebSocket-Key': 'dGhlIHNhbXBsZSBub25jZQ==',
          'Sec-WebSocket-Version': '13',
        },
      },
      (res) => {
        res.resume();
        resolve(res.statusCode);
      }
    );
    req.on('error', reject);
    req.end();
  });
  assert.equal(badOriginStatus, 403, 'Expected 403 for untrusted WebSocket Origin');

  // 4. Confidence < 0.65 gating check via POST /api/telemetry
  const lowConfRes = await fetch(`${BASE_HTTP}/api/telemetry`, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      Origin: TRUSTED_ORIGIN,
    },
    body: JSON.stringify({
      nodeId: 'e2e-low-conf',
      timestamp: Date.now(),
      location: NODE_ALPHA,
      detection: {
        class: 'Shahed_Piston_UAV',
        confidence: 0.4,
        snr_db: 5.0,
        peak_freq_hz: 92.5,
      },
    }),
  });
  assert.equal(lowConfRes.status, 202);
  const lowConfBody = await lowConfRes.json();
  assert.equal(lowConfBody.accepted, false);

  // 5. Live multi-node WebSocket track fusion across SpatialZoneDO
  const urlA = `${BASE_WS}/ws?lat=${NODE_ALPHA.lat}&lng=${NODE_ALPHA.lng}&nodeId=${NODE_ALPHA.nodeId}`;
  const urlB = `${BASE_WS}/ws?lat=${NODE_BETA.lat}&lng=${NODE_BETA.lng}&nodeId=${NODE_BETA.nodeId}`;

  const clientA = await openWsWithOrigin(urlA, TRUSTED_ORIGIN);
  const clientB = await openWsWithOrigin(urlB, TRUSTED_ORIGIN);

  try {
    const snapA = await clientA.waitFor((m) => m.type === 'snapshot');
    const snapB = await clientB.waitFor((m) => m.type === 'snapshot');
    assert.equal(snapA.cellId, cellData.cellId);
    assert.equal(snapB.cellId, cellData.cellId);
    assert.equal(snapA.boundary.length, 6);

    const t0 = Date.now();
    clientA.ws.send(
      JSON.stringify({
        nodeId: NODE_ALPHA.nodeId,
        timestamp: t0,
        location: NODE_ALPHA,
        detection: {
          class: 'Shahed_Piston_UAV',
          confidence: 0.94,
          snr_db: 14.0,
          peak_freq_hz: 92.5,
        },
      })
    );

    await new Promise((r) => setTimeout(r, 600));

    const t1 = t0 + 600;
    clientB.ws.send(
      JSON.stringify({
        nodeId: NODE_BETA.nodeId,
        timestamp: t1,
        location: NODE_BETA,
        detection: {
          class: 'Shahed_Piston_UAV',
          confidence: 0.89,
          snr_db: 4.0,
          peak_freq_hz: 93.1,
        },
      })
    );

    const hasFusedTrack = (m) =>
      m.type === 'track_update' &&
      Array.isArray(m.tracks) &&
      m.tracks.some((tr) => tr.class === 'Shahed_Piston_UAV' && tr.nodeCount >= 2);

    const updateA = await clientA.waitFor(hasFusedTrack);
    const updateB = await clientB.waitFor(hasFusedTrack);

    const fusedTrack = updateA.tracks.find(
      (tr) => tr.class === 'Shahed_Piston_UAV' && tr.nodeCount >= 2
    );
    assert.ok(fusedTrack, 'Expected fused Shahed_Piston_UAV track with nodeCount >= 2');
    assert.equal(updateB.cellId, cellData.cellId);

    const midpointLng = (NODE_ALPHA.lng + NODE_BETA.lng) / 2;
    assert.ok(
      fusedTrack.lng < midpointLng,
      `Expected SNR-weighted lng (${fusedTrack.lng}) closer to alpha than midpoint (${midpointLng})`
    );
    assert.ok(Number.isFinite(fusedTrack.vx));
    assert.ok(Number.isFinite(fusedTrack.vy));
    assert.ok(Number.isFinite(fusedTrack.speedMps));

    // Verify DO persisted state endpoint
    const stateRes = await fetch(`${BASE_HTTP}/api/zone/${cellData.cellId}/state`);
    assert.equal(stateRes.status, 200);
    const stateData = await stateRes.json();
    assert.equal(stateData.cellId, cellData.cellId);
    assert.ok(stateData.tracks.length >= 1);
    assert.ok(stateData.nodes.length >= 2);

    console.log(
      JSON.stringify(
        {
          status: 'PASS',
          cellId: cellData.cellId,
          origin403Status: badOriginStatus,
          lowConfidence202Accepted: lowConfBody.accepted,
          fusedTrack: {
            trackId: fusedTrack.trackId,
            class: fusedTrack.class,
            lat: Number(fusedTrack.lat.toFixed(6)),
            lng: Number(fusedTrack.lng.toFixed(6)),
            midpointLng,
            nodeCount: fusedTrack.nodeCount,
            speedMps: Number(fusedTrack.speedMps.toFixed(2)),
            confidence: fusedTrack.confidence,
          },
          persistedZoneNodes: stateData.nodes.length,
        },
        null,
        2
      )
    );
  } finally {
    clientA.ws.close(1000, 'E2E verify done');
    clientB.ws.close(1000, 'E2E verify done');
    await new Promise((r) => setTimeout(r, 100));
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
