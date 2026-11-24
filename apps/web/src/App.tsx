import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { MelSpectrogramExtractor } from './audio/MelSpectrogramExtractor';
import { StreamManager } from './audio/StreamManager';
import { SpatialMap } from './components/SpatialMap';
import { SpectrogramCanvas } from './components/SpectrogramCanvas';
import {
  getDefaultMeshWsUrl,
  getH3CellInfo,
  NodeTelemetryClient,
} from './mesh/NodeTelemetry';
import { InferenceEngine, type EngineDiagnostics } from './ml/InferenceEngine';
import {
  CONFIDENCE_THRESHOLD,
  TRANSMITTABLE_UAV_CLASSES,
  UAV_CLASSES,
  type AudioCaptureState,
  type AudioClass,
  type InferenceResult,
  type MeshConnectionState,
  type MeshZoneState,
  type NodeLocation,
  type SignalMetrics,
  type SimulationProfile,
} from './types';

interface LogItem {
  id: number;
  time: string;
  level: 'info' | 'warn' | 'tx';
  message: string;
}

const DEFAULT_LOCATION: NodeLocation = {
  lat: 50.4501,
  lng: 30.5234,
  accuracy: 25,
};

function getOrCreateNodeId(): string {
  try {
    const existing = localStorage.getItem('panotti_node_id');
    if (existing) return existing;
    const generated = `node-${Math.random().toString(36).slice(2, 8)}`;
    localStorage.setItem('panotti_node_id', generated);
    return generated;
  } catch {
    return `node-${Math.random().toString(36).slice(2, 8)}`;
  }
}

function getBarColor(cls: AudioClass): string {
  switch (cls) {
    case 'Shahed_Piston_UAV':
      return 'var(--accent-danger)';
    case 'Multirotor_UAV':
      return 'var(--accent-warn)';
    case 'FixedWing_Electric_UAV':
      return 'var(--accent-violet)';
    case 'Ground_Combustion_Noise':
      return 'var(--accent-primary)';
    case 'Environmental_Background':
      return 'var(--accent-good)';
  }
}

function formatTime(): string {
  return new Date().toLocaleTimeString([], {
    hour12: false,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

export function App() {
  const [nodeId] = useState<string>(() => getOrCreateNodeId());
  const [location, setLocation] = useState<NodeLocation>(DEFAULT_LOCATION);
  const [latInput, setLatInput] = useState<string>(String(DEFAULT_LOCATION.lat));
  const [lngInput, setLngInput] = useState<string>(String(DEFAULT_LOCATION.lng));
  const [meshUrl, setMeshUrl] = useState<string>(() => getDefaultMeshWsUrl());

  const [audioState, setAudioState] = useState<AudioCaptureState>('idle');
  const [activeSim, setActiveSim] = useState<SimulationProfile | null>(null);
  const [audioRate, setAudioRate] = useState<number>(16000);

  const [spectrogram, setSpectrogram] = useState<Float32Array | null>(null);
  const [metrics, setMetrics] = useState<SignalMetrics>({ snr_db: -30, peak_freq_hz: 0 });
  const [inference, setInference] = useState<InferenceResult | null>(null);
  const [engineDiag, setEngineDiag] = useState<EngineDiagnostics>({
    status: 'uninitialized',
    provider: null,
    crossOriginIsolated:
      typeof globalThis !== 'undefined' &&
      Boolean((globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated),
    configuredThreads: 1,
    wasmPaths: '/ort/',
    message: 'Initializing...',
  });

  const [meshState, setMeshState] = useState<MeshConnectionState>('disconnected');
  const [zoneState, setZoneState] = useState<MeshZoneState | null>(null);
  const [logs, setLogs] = useState<LogItem[]>([]);

  const logCounterRef = useRef(0);
  const extractorRef = useRef<MelSpectrogramExtractor | null>(null);
  const engineRef = useRef<InferenceEngine | null>(null);
  const streamRef = useRef<StreamManager | null>(null);
  const telemetryRef = useRef<NodeTelemetryClient | null>(null);
  const inferringRef = useRef(false);
  const lastSuppressedLogAtRef = useRef(0);

  if (!extractorRef.current) {
    extractorRef.current = new MelSpectrogramExtractor();
  }

  const appendLog = useCallback((level: 'info' | 'warn' | 'tx', message: string) => {
    logCounterRef.current += 1;
    const item: LogItem = {
      id: logCounterRef.current,
      time: formatTime(),
      level,
      message,
    };
    setLogs((prev) => [item, ...prev].slice(0, 60));
  }, []);

  const cellInfo = useMemo(
    () => getH3CellInfo(location.lat, location.lng),
    [location.lat, location.lng]
  );

  const initEngine = useCallback(async () => {
    if (!engineRef.current) {
      engineRef.current = new InferenceEngine();
    }
    const engine = engineRef.current;
    setEngineDiag(engine.getDiagnostics());
    const ok = await engine.init();
    const diag = engine.getDiagnostics();
    setEngineDiag(diag);
    appendLog(ok ? 'info' : 'warn', diag.message);
  }, [appendLog]);

  useEffect(() => {
    void initEngine();

    const telemetry = new NodeTelemetryClient({
      nodeId,
      baseUrl: meshUrl,
      location: DEFAULT_LOCATION,
      onStateChange: (state) => {
        setMeshState(state);
      },
      onZoneUpdate: (zone) => {
        setZoneState(zone);
      },
      onLog: (level, message) => {
        appendLog(level, message);
      },
    });
    telemetryRef.current = telemetry;

    const handleFrame = async (frame16k: Float32Array) => {
      const extractor = extractorRef.current;
      if (!extractor) return;

      const mel = extractor.extract(frame16k);
      const sigMetrics = extractor.estimateSignalMetrics(frame16k);
      setSpectrogram(mel);
      setMetrics(sigMetrics);

      const engine = engineRef.current;
      if (!engine || !engine.isReady() || inferringRef.current) {
        return;
      }

      inferringRef.current = true;
      try {
        const res = await engine.run(mel);
        setInference(res);

        const dispatch = telemetry.reportDetection({
          class: res.topClass,
          confidence: res.confidence,
          snr_db: sigMetrics.snr_db,
          peak_freq_hz: sigMetrics.peak_freq_hz,
        });

        if (dispatch.transmitted) {
          appendLog(
            'tx',
            `TX ${res.topClass} conf=${(res.confidence * 100).toFixed(1)}% peak=${sigMetrics.peak_freq_hz}Hz snr=${sigMetrics.snr_db}dB`
          );
        } else if (
          TRANSMITTABLE_UAV_CLASSES.has(res.topClass) &&
          Date.now() - lastSuppressedLogAtRef.current > 2500
        ) {
          lastSuppressedLogAtRef.current = Date.now();
          appendLog('info', dispatch.reason ?? 'Sub-threshold UAV signature');
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        appendLog('warn', `Inference error: ${msg}`);
      } finally {
        inferringRef.current = false;
      }
    };

    const stream = new StreamManager({
      onFrame: (frame) => {
        void handleFrame(frame);
      },
      onStateChange: (state, sr) => {
        setAudioState(state);
        setAudioRate(sr);
        if (state === 'idle') {
          setActiveSim(null);
        }
      },
      onError: (msg) => {
        appendLog('warn', msg);
      },
    });
    streamRef.current = stream;

    return () => {
      stream.stop();
      telemetry.disconnect();
      void engineRef.current?.dispose();
    };
  }, [appendLog, initEngine, nodeId]);

  const handleStartMic = async () => {
    setActiveSim(null);
    appendLog('info', 'Requesting raw 16 kHz microphone stream (AEC/NS/AGC disabled)...');
    try {
      await streamRef.current?.startMicrophone();
      appendLog('info', 'Microphone AudioWorklet stream active');
    } catch {
      // Error surfaced via StreamManager onError
    }
  };

  const handleSimulate = (profile: SimulationProfile) => {
    setActiveSim(profile);
    appendLog('info', `Started synthetic acoustic simulation: ${profile}`);
    streamRef.current?.startSimulation(profile);
  };

  const handleStop = () => {
    streamRef.current?.stop();
    setActiveSim(null);
    appendLog('info', 'Audio pipeline stopped');
  };

  const handleApplyLocation = () => {
    const lat = Number(latInput);
    const lng = Number(lngInput);
    if (!Number.isFinite(lat) || !Number.isFinite(lng) || lat < -90 || lat > 90 || lng < -180 || lng > 180) {
      appendLog('warn', 'Invalid coordinates: lat must be [-90, 90], lng [-180, 180]');
      return;
    }
    const nextLoc: NodeLocation = { lat, lng, accuracy: 15 };
    setLocation(nextLoc);
    telemetryRef.current?.updateLocation(nextLoc);
    appendLog('info', `Node location updated (${lat.toFixed(4)}, ${lng.toFixed(4)})`);
  };

  const handleGeolocate = () => {
    if (!navigator.geolocation) {
      appendLog('warn', 'Geolocation API unavailable in this browser');
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const nextLoc: NodeLocation = {
          lat: Number(pos.coords.latitude.toFixed(6)),
          lng: Number(pos.coords.longitude.toFixed(6)),
          accuracy: Math.round(pos.coords.accuracy || 20),
        };
        setLatInput(String(nextLoc.lat));
        setLngInput(String(nextLoc.lng));
        setLocation(nextLoc);
        telemetryRef.current?.updateLocation(nextLoc);
        appendLog('info', `GPS lock acquired (±${nextLoc.accuracy} m)`);
      },
      (err) => {
        appendLog('warn', `GPS failed: ${err.message}`);
      },
      { enableHighAccuracy: true, timeout: 8000 }
    );
  };

  const handleToggleMesh = () => {
    if (meshState === 'connected' || meshState === 'connecting' || meshState === 'reconnecting') {
      telemetryRef.current?.disconnect();
      appendLog('info', 'Mesh WebSocket disconnected');
    } else {
      telemetryRef.current?.connect(meshUrl, location);
    }
  };

  const runtimePillClass =
    engineDiag.status === 'ready'
      ? 'pill pill-good'
      : engineDiag.status === 'loading'
        ? 'pill pill-info'
        : 'pill pill-warn';

  const meshPillClass =
    meshState === 'connected'
      ? 'pill pill-good'
      : meshState === 'disconnected'
        ? 'pill'
        : 'pill pill-warn';

  const activeBoundary =
    zoneState && zoneState.cellId === cellInfo.cellId && zoneState.boundary.length >= 3
      ? zoneState.boundary
      : cellInfo.boundary;

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <svg width="32" height="32" viewBox="0 0 64 64" fill="none" aria-hidden="true">
            <rect width="64" height="64" rx="14" fill="#090d12" />
            <circle cx="32" cy="32" r="24" stroke="#22d3ee" strokeOpacity="0.35" strokeWidth="2" />
            <circle cx="32" cy="32" r="15" stroke="#22d3ee" strokeOpacity="0.65" strokeWidth="2" />
            <circle cx="32" cy="32" r="6" fill="#22d3ee" />
            <path d="M32 32 L49 15" stroke="#34d399" strokeWidth="2.5" strokeLinecap="round" />
          </svg>
          <div>
            <h1 className="brand-title">PANOTTI // ACOUSTIC MESH NODE</h1>
            <div className="brand-sub">
              Node {nodeId} · 16 kHz STFT-Mel Edge Classifier · H3 Res 6 Fusion
            </div>
          </div>
        </div>

        <div className="status-pills">
          <span className={`pill ${engineDiag.crossOriginIsolated ? 'pill-good' : 'pill-warn'}`}>
            <span className="pill-dot" />
            COI: {engineDiag.crossOriginIsolated ? 'Isolated' : 'Fallback'}
          </span>

          <span className={runtimePillClass}>
            <span className="pill-dot" />
            ORT:{' '}
            {engineDiag.status === 'ready'
              ? `${engineDiag.provider?.toUpperCase()} (${engineDiag.configuredThreads}T)`
              : engineDiag.status === 'unavailable'
                ? 'Model Pending'
                : engineDiag.status}
          </span>

          <span className={`pill ${audioState === 'idle' ? 'pill' : 'pill-info'}`}>
            <span className="pill-dot" />
            Audio:{' '}
            {audioState === 'idle'
              ? 'Idle'
              : audioState === 'microphone'
                ? `Mic (${audioRate}Hz)`
                : `Sim (${activeSim})`}
          </span>

          <span className={meshPillClass}>
            <span className="pill-dot" />
            Mesh: {meshState}
          </span>
        </div>
      </header>

      {(engineDiag.status === 'unavailable' || engineDiag.status === 'error') && (
        <div className="banner-notice" role="status">
          <span>{engineDiag.message}</span>
          <button type="button" className="btn btn-warn btn-xs" onClick={() => void initEngine()}>
            Retry Model Load
          </button>
        </div>
      )}

      <section className="controls-bar" aria-label="Sensor Controls">
        <div className="action-buttons">
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => void handleStartMic()}
            disabled={audioState === 'microphone'}
          >
            Start Mic
          </button>
          <button
            type="button"
            className={`btn ${activeSim === 'shahed' ? 'btn-danger' : ''}`}
            onClick={() => handleSimulate('shahed')}
          >
            Simulate Shahed-136
          </button>
          <button
            type="button"
            className={`btn ${activeSim === 'multirotor' ? 'btn-warn' : ''}`}
            onClick={() => handleSimulate('multirotor')}
          >
            Simulate Multirotor
          </button>
          <button
            type="button"
            className="btn"
            onClick={() => handleSimulate('background')}
          >
            Simulate Background
          </button>
          <button
            type="button"
            className="btn btn-ghost"
            onClick={handleStop}
            disabled={audioState === 'idle'}
          >
            Stop
          </button>
        </div>

        <div className="node-config-grid">
          <label className="field field-coord">
            <span>LAT</span>
            <input
              type="text"
              value={latInput}
              onChange={(e) => setLatInput(e.target.value)}
              aria-label="Node Latitude"
            />
          </label>
          <label className="field field-coord">
            <span>LNG</span>
            <input
              type="text"
              value={lngInput}
              onChange={(e) => setLngInput(e.target.value)}
              aria-label="Node Longitude"
            />
          </label>
          <button type="button" className="btn btn-xs" onClick={handleApplyLocation}>
            Set Coords
          </button>
          <button type="button" className="btn btn-xs" onClick={handleGeolocate}>
            GPS
          </button>

          <div className="h3-readout">
            <span className="h3-label">H3 CELL (RES 6)</span>
            <span className="h3-value">{cellInfo.cellId}</span>
          </div>

          <label className="field field-url">
            <span>MESH HUB WS URL</span>
            <input
              type="url"
              value={meshUrl}
              onChange={(e) => setMeshUrl(e.target.value)}
              aria-label="Mesh Hub WebSocket URL"
            />
          </label>
          <button
            type="button"
            className={`btn ${meshState === 'connected' ? 'btn-danger' : 'btn-primary'}`}
            onClick={handleToggleMesh}
          >
            {meshState === 'disconnected' ? 'Connect Mesh' : 'Disconnect'}
          </button>
        </div>
      </section>

      <main className="workspace-grid">
        <div className="left-column">
          <SpectrogramCanvas
            spectrogram={spectrogram}
            peakFreqHz={metrics.peak_freq_hz}
            activeClass={inference?.topClass ?? null}
            confidence={inference?.confidence ?? 0}
          />

          <div className="metrics-strip">
            <div className="metric-box">
              <span className="metric-label">PEAK FREQ</span>
              <span className="metric-val">{metrics.peak_freq_hz.toFixed(1)} Hz</span>
            </div>
            <div className="metric-box">
              <span className="metric-label">BAND SNR</span>
              <span className="metric-val">{metrics.snr_db.toFixed(1)} dB</span>
            </div>
            <div className="metric-box">
              <span className="metric-label">INFERENCE</span>
              <span className="metric-val">
                {inference ? `${inference.inferenceMs.toFixed(1)} ms` : '—'}
              </span>
            </div>
            <div className="metric-box">
              <span className="metric-label">MESH GATE</span>
              <span className="metric-val">≥ {(CONFIDENCE_THRESHOLD * 100).toFixed(0)}% UAV</span>
            </div>
          </div>

          <section className="card" aria-label="Class Probabilities">
            <div className="panel-title">5-CLASS ACOUSTIC POSTERIORS (LINE = 65% MESH GATE)</div>
            <div className="meters-list">
              {UAV_CLASSES.map((cls) => {
                const prob = inference?.probabilities[cls] ?? 0;
                const pct = Math.max(0, Math.min(100, prob * 100));
                return (
                  <div key={cls} className="meter-row">
                    <div className="meter-label-row">
                      <span>{cls}</span>
                      <span>{pct.toFixed(1)}%</span>
                    </div>
                    <div className="meter-track">
                      <div
                        className="meter-fill"
                        style={{
                          width: `${pct}%`,
                          background: getBarColor(cls),
                        }}
                      />
                      <div className="meter-threshold" title="0.65 Telemetry Transmission Gate" />
                    </div>
                  </div>
                );
              })}
            </div>
          </section>
        </div>

        <div className="right-column">
          <SpatialMap
            localNodeId={nodeId}
            localLocation={location}
            cellId={cellInfo.cellId}
            boundary={activeBoundary}
            nodes={zoneState?.nodes ?? []}
            tracks={zoneState?.tracks ?? []}
          />

          <section className="card" aria-label="Telemetry Event Log">
            <div className="panel-title">
              TELEMETRY & MESH LOG ({zoneState?.tracks.length ?? 0} ACTIVE TRACKS ·{' '}
              {zoneState?.nodes.length ?? 0} ZONE NODES)
            </div>
            <div className="log-list scroll-region">
              {logs.length === 0 ? (
                <div className="log-entry log-entry-info">
                  <span className="log-time">--:--:--</span>
                  <span>Ready. Start Mic or launch a synthetic drone wave.</span>
                </div>
              ) : (
                logs.map((entry) => (
                  <div key={entry.id} className={`log-entry log-entry-${entry.level}`}>
                    <span className="log-time">{entry.time}</span>
                    <span>{entry.message}</span>
                  </div>
                ))
              )}
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}
export default App;
