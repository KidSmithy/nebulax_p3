import { create } from 'zustand';
import { DEFAULT_CAR, LINE_IDS } from '../lib/lines';
import {
  AIAnswer,
  AIInsight,
  ActiveInterventions,
  CameraPreset,
  Finding,
  Line,
  LogEntry,
  SubsystemSelection,
  UiMode,
  UnifiedTelemetryFrame,
  WhatIfResult,
} from '../types/telemetry';

/** Tagged onto each socket so its own onclose knows whether it was closed on purpose. */
type TaggedSocket = WebSocket & { _manualClose?: boolean };

const EMPTY_INTERVENTIONS: ActiveInterventions = {
  ACTION_GRIND_RAIL: false,
  ACTION_LUBRICATE_DOOR: false,
  ACTION_REPLACE_FILTER: false,
  ACTION_INSPECT_BEARING: false,
};

export const CAR_COUNT = 8;

type ConductorState = 'docked' | 'popup' | 'expanded' | 'onboarding';

/**
 * Everything that belongs to one line's own Conductor and view. The active
 * line's values live on the store's top-level fields (so components read them
 * unchanged); only the inactive line is parked here.
 */
interface LineSnapshot {
  monitoredCar: number;
  conductorState: ConductorState;
  aiAnswers: AIAnswer[];
  uploadResult: Finding | null;
  selectedSubsystem: SubsystemSelection;
}

const freshSnapshot = (): LineSnapshot => ({
  monitoredCar: DEFAULT_CAR,
  conductorState: 'docked',
  aiAnswers: [],
  uploadResult: null,
  selectedSubsystem: 'overview',
});
/** Which ACV finding each line's monitored car was last synced to (see setFrame). */
const syncedAcvKey: Record<Line, string> = { NSL: '', EWL: '' };
const ZOOM_FOR_MODE: Record<CameraPreset, number> = { micro: 0, meso: 0.5, macro: 1 };

interface TwinState {
  currentFrame: UnifiedTelemetryFrame | null;
  history: UnifiedTelemetryFrame[];
  cameraMode: CameraPreset;
  /** 0 = close on the selected component, 0.5 = one car, 1 = all 8 cars. */
  cameraZoom: number;
  /** Which line the scene, findings and Conductor currently belong to. */
  activeLine: Line;
  /** The other line's parked state (see LineSnapshot). */
  lineSnapshots: Record<Line, LineSnapshot>;
  /** Which of the 8 cars (1-8) the scene focuses on and hangs its hotspots on. */
  monitoredCar: number;
  /** The upload's own direct response, for the onboarding walkthrough's results step. */
  uploadResult: Finding | null;
  uploading: boolean;
  uploadError: string | null;
  /** Persistent (backend, in-memory) receipt log of every resolved finding. */
  resolvedLog: LogEntry[];
  selectedSubsystem: SubsystemSelection;
  xrayMode: boolean;
  isPlaying: boolean;
  playbackSpeed: number;
  isConnected: boolean;
  activeInterventions: ActiveInterventions;
  ws: WebSocket | null;

  // UI Decluttering & Layout Controls
  isHudVisible: boolean;
  isLeftDrawerOpen: boolean;
  isRightDrawerOpen: boolean;
  /** The docked panel that tabulates the active line's uploaded results. */
  resultsPanelOpen: boolean;
  rightDrawerTab: 'telemetry' | 'whatif' | 'ai';
  activeInspection: 'door' | 'acv' | 'shm' | 'rail' | null;

  /** Beginner leads with plain language; Expert exposes raw engineering units. */
  uiMode: UiMode;

  /** Last measured impact echoed back by the backend after a toggle. */
  lastWhatIfResult: WhatIfResult | null;

  // AI assistant
  aiInsight: AIInsight | null;
  aiInsightLoading: boolean;
  aiAutoRefresh: boolean;
  aiEnabled: boolean | null;
  aiModel: string | null;
  aiAnswers: AIAnswer[];
  aiAsking: boolean;

  // Conductor assistant - new surface, gated behind conductorEnabled.
  conductorEnabled: boolean;
  conductorState: ConductorState;

  // Actions
  setFrame: (frame: UnifiedTelemetryFrame) => void;
  setCameraMode: (mode: CameraPreset) => void;
  setCameraZoom: (zoom: number) => void;
  setActiveLine: (line: Line) => void;
  setMonitoredCar: (car: number) => void;
  uploadSubsystemData: (subsystem: SubsystemSelection, file: File) => Promise<Finding | null>;
  fetchLog: () => Promise<void>;
  /** Puts the backend back to its first-load findings: 4 bubbles on NSL, none on EWL. */
  resetToSeededFindings: () => Promise<void>;
  resolveFinding: (subsystem: SubsystemSelection) => Promise<void>;
  setSelectedSubsystem: (subsystem: SubsystemSelection) => void;
  setActiveInspection: (inspection: 'door' | 'acv' | 'shm' | 'rail' | null) => void;
  toggleXray: () => void;
  toggleHud: () => void;
  toggleLeftDrawer: () => void;
  toggleRightDrawer: () => void;
  setResultsPanelOpen: (open: boolean) => void;
  setRightDrawerTab: (tab: 'telemetry' | 'whatif' | 'ai') => void;
  setUiMode: (mode: UiMode) => void;
  toggleUiMode: () => void;
  setPlaying: (playing: boolean) => void;
  setPlaybackSpeed: (speed: number) => void;
  setIntervention: (action: string, enabled: boolean) => void;
  setConnected: (connected: boolean) => void;
  initWebSocket: () => void;
  disconnectWebSocket: () => void;
  sendSeek: (kp: number) => void;
  triggerWhatIf: (action: string, enabled: boolean) => void;
  resetWhatIf: () => void;
  dismissWhatIfResult: () => void;
  fetchAiStatus: () => Promise<void>;
  fetchAiInsight: (force?: boolean) => Promise<void>;
  setAiAutoRefresh: (on: boolean) => void;
  askAi: (question: string) => Promise<void>;
  setConductorState: (s: 'docked' | 'popup' | 'expanded' | 'onboarding') => void;
  setConductorEnabled: (on: boolean) => void;
}

export const useTwinStore = create<TwinState>((set, get) => {
  /** Write to a line's own state whether it is the active line or parked - async work may finish after a tab switch. */
  const patchLine = (line: Line, patch: Partial<LineSnapshot>) => {
    if (line === get().activeLine) set(patch);
    else
      set((state) => ({
        lineSnapshots: { ...state.lineSnapshots, [line]: { ...state.lineSnapshots[line], ...patch } },
      }));
  };
  const readLine = (line: Line): LineSnapshot =>
    line === get().activeLine ? (get() as unknown as LineSnapshot) : get().lineSnapshots[line];

  return {
  currentFrame: null,
  history: [],
  cameraMode: 'meso',
  cameraZoom: ZOOM_FOR_MODE.meso,
  activeLine: 'NSL',
  lineSnapshots: { NSL: freshSnapshot(), EWL: freshSnapshot() },
  monitoredCar: DEFAULT_CAR,
  uploadResult: null,
  uploading: false,
  uploadError: null,
  resolvedLog: [],
  selectedSubsystem: 'overview',
  xrayMode: false,
  isPlaying: true,
  playbackSpeed: 1.0,
  isConnected: false,
  activeInterventions: { ...EMPTY_INTERVENTIONS },
  ws: null,

  // UI state defaults (right drawer starts closed for clean spacious 3D view)
  isHudVisible: true,
  isLeftDrawerOpen: true,
  isRightDrawerOpen: false,
  resultsPanelOpen: false,
  rightDrawerTab: 'telemetry',
  activeInspection: null,

  // Default to beginner: the dashboard should be readable before it is dense.
  uiMode: 'beginner',

  lastWhatIfResult: null,

  aiInsight: null,
  aiInsightLoading: false,
  aiAutoRefresh: false,
  aiEnabled: null,
  aiModel: null,
  aiAnswers: [],
  aiAsking: false,

  conductorEnabled: true,
  conductorState: 'docked',

  setFrame: (frame) =>
    set((state) => {
      const newHistory = [...state.history, frame].slice(-60); // keep last 60 frames (6s at 10Hz)

      // Each line's unresolved ACV finding names the car everything else on that line
      // hangs on. Follow it whenever a new one is seen - including right after a page
      // reload, when the backend still holds the finding but the client has fallen back
      // to the default car.
      let monitoredCar = state.monitoredCar;
      let lineSnapshots = state.lineSnapshots;
      for (const line of LINE_IDS) {
        const acv = frame.uploads_by_line?.[line]?.acv;
        const acvKey = acv?.most_likely_faulty_car ? `${acv.file_name}|${acv.most_likely_faulty_car}` : '';
        if (acvKey === syncedAcvKey[line]) continue;
        syncedAcvKey[line] = acvKey;
        if (!acvKey) continue;
        const car = Math.min(CAR_COUNT, Math.max(1, Math.round(Number(acv!.most_likely_faulty_car))));
        if (line === state.activeLine) monitoredCar = car;
        else lineSnapshots = { ...lineSnapshots, [line]: { ...lineSnapshots[line], monitoredCar: car } };
      }

      return {
        currentFrame: frame,
        history: newHistory,
        monitoredCar,
        lineSnapshots,
        // Trust the backend as the source of truth for which actions are live,
        // so the UI cannot drift out of sync with the simulation.
        activeInterventions: frame.active_interventions ?? state.activeInterventions,
      };
    }),

  setCameraMode: (mode) => set({ cameraMode: mode, cameraZoom: ZOOM_FOR_MODE[mode] }),

  setCameraZoom: (zoom) => {
    const z = Math.min(1, Math.max(0, zoom));
    set({ cameraZoom: z, cameraMode: z < 0.25 ? 'micro' : z < 0.75 ? 'meso' : 'macro' });
  },

  setActiveLine: (line) =>
    set((state) => {
      if (line === state.activeLine) return {};
      const parked: LineSnapshot = {
        monitoredCar: state.monitoredCar,
        conductorState: state.conductorState,
        aiAnswers: state.aiAnswers,
        uploadResult: state.uploadResult,
        selectedSubsystem: state.selectedSubsystem,
      };
      const next = state.lineSnapshots[line];
      return {
        activeLine: line,
        lineSnapshots: { ...state.lineSnapshots, [state.activeLine]: parked },
        ...next,
        activeInspection: null,
        aiInsight: null,
        uploadError: null,
      };
    }),

  setMonitoredCar: (car) => set({ monitoredCar: Math.min(CAR_COUNT, Math.max(1, Math.round(car))) }),

  uploadSubsystemData: async (subsystem, file) => {
    const line = get().activeLine;
    set({ uploading: true, uploadError: null });
    // Firebase Hosting rewrites have a hard 32 MB payload limit.
    const MAX_SIZE_MB = 30;
    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
      set({
        uploading: false,
        uploadError: `File is too large (${(file.size / (1024 * 1024)).toFixed(1)} MB). Firebase Hosting limits uploads to ${MAX_SIZE_MB} MB. Please upload a single test file (e.g., Test1.csv, ~17 MB) rather than the entire Rail_Test.zip archive.`,
      });
      return null;
    }

    try {
      const body = new FormData();
      body.append('file', file);
      body.append('subsystem', subsystem);
      body.append('line', line);
      const res = await fetch('/api/predict/upload', { method: 'POST', body });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        let msg = typeof data.detail === 'string' ? data.detail : null;
        if (!msg) {
          if (res.status === 413) {
            msg = 'File is too large for the hosting proxy (HTTP 413). Please upload a file under 30 MB.';
          } else if (res.status === 504) {
            msg = 'The upload timed out (HTTP 504). Please try a smaller single-run file.';
          } else {
            msg = 'The upload failed. Try again.';
          }
        }
        set({ uploadError: msg });
        return null;
      }
      const finding = data as Finding;
      patchLine(line, { uploadResult: finding });
      set({ resultsPanelOpen: true }); // a fresh upload's results are the thing to look at next
      // Establishes which car everything else gets tagged to, per the "ACV
      // resolves first" assumption - see useMonitoredItems/TrainAssembly.
      if (subsystem === 'acv' && finding.most_likely_faulty_car) {
        patchLine(line, {
          monitoredCar: Math.min(CAR_COUNT, Math.max(1, Math.round(Number(finding.most_likely_faulty_car)))),
        });
      }
      return finding;
    } catch {
      set({ uploadError: 'Could not reach the backend. Check that it is running.' });
      return null;
    } finally {
      set({ uploading: false });
    }
  },

  fetchLog: async () => {
    try {
      const res = await fetch('/api/log');
      const data = await res.json();
      set({ resolvedLog: data.log ?? [] });
    } catch (err) {
      console.error('[Log] fetch failed:', err);
    }
  },

  // Findings live on the backend, so they would otherwise survive a refresh in
  // whatever state the last session left them (bubbles resolved away, files
  // uploaded onto EWL). Every page load re-asserts the intended demo start:
  // the four seeded inspection tags on NSL, and a clean EWL for the upload
  // walkthrough. Safe to fail - a refresh that cannot reach the backend just
  // keeps the existing server state instead of blocking the app.
  resetToSeededFindings: async () => {
    try {
      await fetch('/api/predict/seed?reset=1', { method: 'POST' });
      // The monitored car is re-derived from the (re-seeded) ACV finding on the
      // next frame, so forget what the previous frames synced to.
      for (const line of LINE_IDS) syncedAcvKey[line] = '';
    } catch (err) {
      console.warn('[Seed] Could not restore the seeded findings:', err);
    }
  },

  resolveFinding: async (subsystem) => {
    try {
      const res = await fetch('/api/predict/resolve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ subsystem, car: get().monitoredCar, line: get().activeLine }),
      });
      const entry = await res.json().catch(() => null);
      if (res.ok && entry) {
        set((state) => ({ resolvedLog: [...state.resolvedLog, entry as LogEntry] }));
      }
    } catch (err) {
      console.error('[Log] resolve failed:', err);
    }
  },

  setSelectedSubsystem: (subsystem) => {
    set({ selectedSubsystem: subsystem });
  },

  setActiveInspection: (inspection) => {
    if (inspection) {
      // The card only exists for an unresolved uploaded finding, so clicking a
      // bare part just selects the subsystem instead of opening a dead card.
      const hasFinding = Boolean(get().currentFrame?.uploads_by_line?.[get().activeLine]?.[inspection]);
      set({ activeInspection: hasFinding ? inspection : null, selectedSubsystem: inspection });
    } else {
      set({ activeInspection: null });
    }
  },

  toggleXray: () => set((state) => ({ xrayMode: !state.xrayMode })),

  toggleHud: () => set((state) => ({ isHudVisible: !state.isHudVisible })),
  toggleLeftDrawer: () => set((state) => ({ isLeftDrawerOpen: !state.isLeftDrawerOpen })),
  toggleRightDrawer: () => set((state) => ({ isRightDrawerOpen: !state.isRightDrawerOpen })),
  setResultsPanelOpen: (open) => set({ resultsPanelOpen: open }),
  setRightDrawerTab: (tab) => set({ rightDrawerTab: tab }),

  setUiMode: (mode) => set({ uiMode: mode }),
  toggleUiMode: () =>
    set((state) => ({ uiMode: state.uiMode === 'beginner' ? 'expert' : 'beginner' })),

  setPlaying: (playing) => {
    set({ isPlaying: playing });
    const ws = get().ws;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'playback', playing, speed: get().playbackSpeed }));
    }
  },

  setPlaybackSpeed: (speed) => {
    set({ playbackSpeed: speed });
    const ws = get().ws;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'playback', playing: get().isPlaying, speed }));
    }
  },

  setIntervention: (action, enabled) =>
    set((state) => ({
      activeInterventions: {
        ...state.activeInterventions,
        [action]: enabled,
      },
    })),

  setConnected: (connected) => set({ isConnected: connected }),

  sendSeek: (kp: number) => {
    const ws = get().ws;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'seek', kp }));
    }
  },

  triggerWhatIf: (action, enabled) => {
    get().setIntervention(action, enabled);
    const ws = get().ws;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'whatif', action, enabled }));
    }
  },

  resetWhatIf: () => {
    set({ activeInterventions: { ...EMPTY_INTERVENTIONS }, lastWhatIfResult: null });
    const ws = get().ws;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'reset_whatif' }));
    }
  },

  dismissWhatIfResult: () => set({ lastWhatIfResult: null }),

  // ------------------------------------------------------------------ AI
  fetchAiStatus: async () => {
    try {
      const res = await fetch('/api/ai/status');
      const data = await res.json();
      set({ aiEnabled: !!data.ai_enabled, aiModel: data.model ?? null });
    } catch {
      set({ aiEnabled: false });
    }
  },

  fetchAiInsight: async (force = false) => {
    if (get().aiInsightLoading) return;
    set({ aiInsightLoading: true });
    try {
      const res = await fetch('/api/ai/insight', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ force, line: get().activeLine, car: get().monitoredCar }),
      });
      const data = await res.json();
      set({ aiInsight: data.insight ?? null });
    } catch (err) {
      console.error('[AI] insight request failed:', err);
    } finally {
      set({ aiInsightLoading: false });
    }
  },

  setAiAutoRefresh: (on) => set({ aiAutoRefresh: on }),

  askAi: async (question) => {
    const q = question.trim();
    if (!q || get().aiAsking) return;
    // The answer belongs to the line it was asked on, even if the tab changes meanwhile.
    const line = get().activeLine;
    const { monitoredCar } = readLine(line);
    const append = (item: AIAnswer) =>
      patchLine(line, { aiAnswers: [...readLine(line).aiAnswers, item].slice(-8) });
    set({ aiAsking: true });
    try {
      const res = await fetch('/api/ai/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, line, car: monitoredCar }),
      });
      const data = await res.json();
      append({ ...data, question: q });
    } catch (err) {
      console.error('[AI] ask request failed:', err);
      append({
        question: q,
        answer: 'Could not reach the assistant. Check that the backend is running.',
        source: 'error',
      });
    } finally {
      set({ aiAsking: false });
    }
  },

  setConductorState: (s) => set({ conductorState: s }),
  setConductorEnabled: (on) => set({ conductorEnabled: on, conductorState: 'docked' }),

  initWebSocket: () => {
    const existing = get().ws;
    if (existing && (existing.readyState === WebSocket.OPEN || existing.readyState === WebSocket.CONNECTING)) {
      return;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.hostname || 'localhost';
    const isLocal = host === 'localhost' || host === '127.0.0.1';
    const wsUrl = import.meta.env.VITE_WS_URL
      || (isLocal
        ? `${protocol}//${host}:8000/ws/telemetry`
        : `wss://smrt-digital-twin-backend-622102147707.asia-southeast1.run.app/ws/telemetry`);

    try {
      const socket = new WebSocket(wsUrl) as TaggedSocket;
      // Tracked from creation (not just on open) so a second initWebSocket call -
      // e.g. React StrictMode's mount->cleanup->mount in dev - sees this one
      // as already CONNECTING instead of opening a duplicate.
      set({ ws: socket });

      socket.onopen = () => {
        set({ isConnected: true });
        console.log('[WebSocket] Connected to digital twin telemetry stream');
      };

      socket.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);

          // The socket now carries two message kinds: continuous telemetry
          // frames, and one-off confirmations of a maintenance intervention
          // carrying its measured impact.
          if (msg.type === 'whatif_result') {
            set({
              lastWhatIfResult: msg as WhatIfResult,
              activeInterventions: msg.active_interventions ?? get().activeInterventions,
            });
            return;
          }

          get().setFrame(msg as UnifiedTelemetryFrame);
        } catch (err) {
          console.error('[WebSocket] Error parsing message:', err);
        }
      };

      socket.onclose = () => {
        if (get().ws === socket) set({ isConnected: false, ws: null });
        if (socket._manualClose) return;
        console.warn('[WebSocket] Stream disconnected. Retrying in 2s...');
        setTimeout(() => get().initWebSocket(), 2000);
      };

      socket.onerror = (error) => {
        console.error('[WebSocket] Stream error:', error);
      };
    } catch (err) {
      console.error('[WebSocket] Initialization error:', err);
    }
  },

  disconnectWebSocket: () => {
    const socket = get().ws as TaggedSocket | null;
    if (!socket) return;
    socket._manualClose = true;
    socket.close();
  },
  };
});
