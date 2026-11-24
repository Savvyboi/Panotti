import { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import type { AudioClass, MeshNodeInfo, NodeLocation, TrackState } from '../types';

export interface SpatialMapProps {
  localNodeId: string;
  localLocation: NodeLocation;
  cellId: string;
  boundary: [number, number][];
  nodes: MeshNodeInfo[];
  tracks: TrackState[];
}

const DARK_TILE_URL = 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png';
const LIGHT_TILE_URL = 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png';

function appendKvRow(parent: HTMLElement, label: string, value: string): void {
  const row = document.createElement('div');
  row.className = 'map-popup-row';

  const keyEl = document.createElement('span');
  keyEl.className = 'map-popup-key';
  keyEl.textContent = label;

  const valEl = document.createElement('span');
  valEl.className = 'map-popup-val';
  valEl.textContent = value;

  row.appendChild(keyEl);
  row.appendChild(valEl);
  parent.appendChild(row);
}

/**
 * Builds a Leaflet popup DOM tree strictly via createElement + textContent
 * (never innerHTML) to eliminate XSS risk from remote mesh telemetry strings.
 */
export function buildNodePopupElement(
  title: string,
  nodeId: string,
  lat: number,
  lng: number,
  cellId: string,
  accuracy?: number
): HTMLElement {
  const container = document.createElement('div');
  container.className = 'map-popup-card';

  const header = document.createElement('div');
  header.className = 'map-popup-title';
  header.textContent = title;
  container.appendChild(header);

  appendKvRow(container, 'Node ID', nodeId);
  appendKvRow(container, 'H3 Cell', cellId);
  appendKvRow(container, 'Coords', `${lat.toFixed(5)}, ${lng.toFixed(5)}`);
  if (typeof accuracy === 'number' && Number.isFinite(accuracy)) {
    appendKvRow(container, 'Accuracy', `±${Math.round(accuracy)} m`);
  }

  return container;
}

export function buildTrackPopupElement(track: TrackState): HTMLElement {
  const container = document.createElement('div');
  container.className = 'map-popup-card';

  const header = document.createElement('div');
  header.className = 'map-popup-title map-popup-title-alert';
  header.textContent = track.class;
  container.appendChild(header);

  appendKvRow(container, 'Track ID', track.trackId);
  appendKvRow(container, 'Confidence', `${(track.confidence * 100).toFixed(1)}%`);
  appendKvRow(container, 'Speed', `${track.speedMps.toFixed(1)} m/s`);
  appendKvRow(container, 'Heading', `${Math.round(track.headingDeg)}°`);
  appendKvRow(container, 'SNR', `${track.snrDb.toFixed(1)} dB`);
  appendKvRow(container, 'Sensors', String(track.nodeCount));
  appendKvRow(container, 'Coords', `${track.lat.toFixed(5)}, ${track.lng.toFixed(5)}`);

  return container;
}

function createLocalNodeIcon(): L.DivIcon {
  return L.divIcon({
    className: 'panotti-div-icon',
    iconSize: [28, 28],
    iconAnchor: [14, 14],
    html: `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28" aria-hidden="true">
      <circle cx="14" cy="14" r="12" fill="rgba(34,211,238,0.18)" stroke="#22d3ee" stroke-width="1.5"/>
      <circle cx="14" cy="14" r="5" fill="#22d3ee"/>
    </svg>`,
  });
}

function createPeerNodeIcon(): L.DivIcon {
  return L.divIcon({
    className: 'panotti-div-icon',
    iconSize: [20, 20],
    iconAnchor: [10, 10],
    html: `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20" aria-hidden="true">
      <circle cx="10" cy="10" r="7.5" fill="rgba(52,211,153,0.22)" stroke="#34d399" stroke-width="1.5"/>
      <circle cx="10" cy="10" r="3" fill="#34d399"/>
    </svg>`,
  });
}

function getTrackColor(uavClass: AudioClass): string {
  if (uavClass === 'Shahed_Piston_UAV') return '#f43f5e';
  if (uavClass === 'Multirotor_UAV') return '#f59e0b';
  return '#a855f7';
}

function createTrackIcon(uavClass: AudioClass, headingDeg: number): L.DivIcon {
  const safeDeg = Number.isFinite(headingDeg) ? Math.round(headingDeg) % 360 : 0;
  const color = getTrackColor(uavClass);
  return L.divIcon({
    className: 'panotti-div-icon',
    iconSize: [32, 32],
    iconAnchor: [16, 16],
    html: `<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 32 32" aria-hidden="true">
      <circle cx="16" cy="16" r="14" fill="rgba(15,23,42,0.75)" stroke="${color}" stroke-width="1.5" stroke-dasharray="3 2"/>
      <g transform="rotate(${safeDeg} 16 16)">
        <polygon points="16,5 23,23 16,19 9,23" fill="${color}"/>
      </g>
    </svg>`,
  });
}

export function SpatialMap({
  localNodeId,
  localLocation,
  cellId,
  boundary,
  nodes,
  tracks,
}: SpatialMapProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<L.Map | null>(null);
  const tileLayerRef = useRef<L.TileLayer | null>(null);
  const overlayLayerRef = useRef<L.LayerGroup | null>(null);
  const [tilesOffline, setTilesOffline] = useState(false);

  useEffect(() => {
    const el = containerRef.current;
    if (!el || mapRef.current) return;

    const map = L.map(el, {
      center: [localLocation.lat, localLocation.lng],
      zoom: 12,
      zoomControl: true,
      attributionControl: true,
    });

    const mql =
      typeof window !== 'undefined' && typeof window.matchMedia === 'function'
        ? window.matchMedia('(prefers-color-scheme: light)')
        : null;
    const initialUrl = mql?.matches ? LIGHT_TILE_URL : DARK_TILE_URL;

    let errorStreak = 0;
    const tileLayer = L.tileLayer(initialUrl, {
      attribution: '&copy; OpenStreetMap &copy; CARTO',
      subdomains: 'abcd',
      maxZoom: 19,
      crossOrigin: 'anonymous',
    });

    tileLayer.on('tileload', () => {
      errorStreak = 0;
      setTilesOffline(false);
    });

    tileLayer.on('tileerror', () => {
      errorStreak += 1;
      if (errorStreak >= 3) {
        setTilesOffline(true);
      }
    });

    tileLayer.addTo(map);

    const overlayGroup = L.layerGroup().addTo(map);

    const handleThemeChange = (ev: MediaQueryListEvent) => {
      errorStreak = 0;
      tileLayer.setUrl(ev.matches ? LIGHT_TILE_URL : DARK_TILE_URL);
    };
    mql?.addEventListener('change', handleThemeChange);

    mapRef.current = map;
    tileLayerRef.current = tileLayer;
    overlayLayerRef.current = overlayGroup;

    return () => {
      mql?.removeEventListener('change', handleThemeChange);
      map.remove();
      mapRef.current = null;
      tileLayerRef.current = null;
      overlayLayerRef.current = null;
    };
  }, []);

  useEffect(() => {
    const overlay = overlayLayerRef.current;
    if (!overlay) return;
    overlay.clearLayers();

    // 1. H3 Resolution 6 hexagon boundary
    if (boundary.length >= 3) {
      L.polygon(boundary, {
        color: '#22d3ee',
        weight: 1.75,
        opacity: 0.75,
        fillColor: '#0ea5e9',
        fillOpacity: 0.06,
        dashArray: '6 4',
      }).addTo(overlay);
    }

    // 2. Peer nodes from mesh hub
    for (const peer of nodes) {
      if (peer.nodeId === localNodeId) continue;
      const marker = L.marker([peer.lat, peer.lng], {
        icon: createPeerNodeIcon(),
      });
      marker.bindPopup(() =>
        buildNodePopupElement('Peer Mesh Node', peer.nodeId, peer.lat, peer.lng, cellId, peer.accuracy)
      );
      marker.addTo(overlay);
    }

    // 3. Local acoustic sensor node
    const localMarker = L.marker([localLocation.lat, localLocation.lng], {
      icon: createLocalNodeIcon(),
      zIndexOffset: 200,
    });
    localMarker.bindPopup(() =>
      buildNodePopupElement(
        'Local Acoustic Sensor',
        localNodeId,
        localLocation.lat,
        localLocation.lng,
        cellId,
        localLocation.accuracy
      )
    );
    localMarker.addTo(overlay);

    // 4. Kalman-filtered UAV tracks + history trails
    for (const track of tracks) {
      const color = getTrackColor(track.class);
      if (track.history.length >= 2) {
        const trail = track.history.map((pt) => [pt.lat, pt.lng] as [number, number]);
        L.polyline(trail, {
          color,
          weight: 2.5,
          opacity: 0.85,
          dashArray: '4 4',
        }).addTo(overlay);
      }

      const trackMarker = L.marker([track.lat, track.lng], {
        icon: createTrackIcon(track.class, track.headingDeg),
        zIndexOffset: 500,
      });
      trackMarker.bindPopup(() => buildTrackPopupElement(track));
      trackMarker.addTo(overlay);
    }
  }, [localNodeId, localLocation, cellId, boundary, nodes, tracks]);

  const handleRecenter = () => {
    mapRef.current?.setView([localLocation.lat, localLocation.lng], 12, { animate: true });
  };

  return (
    <div className="map-panel">
      <div className="map-header">
        <span className="panel-title">SPATIAL MESH RADAR · H3 RES 6 ({cellId})</span>
        <div className="map-header-actions">
          {tilesOffline && (
            <span className="map-offline-pill">Offline Grid Mode · Vector Overlay Active</span>
          )}
          <button type="button" className="btn btn-ghost btn-xs" onClick={handleRecenter}>
            Center Node
          </button>
        </div>
      </div>
      <div className={`map-viewport-wrap ${tilesOffline ? 'map-offline-grid' : ''}`}>
        <div ref={containerRef} className="map-viewport" aria-label="Tactical H3 Acoustic Map" />
      </div>
    </div>
  );
}
