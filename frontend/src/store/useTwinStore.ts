import { create } from 'zustand';
import {
  AIAnswer,
  AIInsight,
  ActiveInterventions,
  CameraPreset,
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

interface TwinState {
  currentFrame: UnifiedTelemetryFrame | null;
  history: UnifiedTelemetryFrame[];
  cameraMode: CameraPreset;
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

  // Actions
  setFrame: (frame: UnifiedTelemetryFrame) => void;
  setCameraMode: (mode: CameraPreset) => void;
  setSelectedSubsystem: (subsystem: SubsystemSelection) => void;
  setActiveInspection: (inspection: 'door' | 'acv' | 'shm' | 'rail' | null) => void;
  toggleXray: () => void;
  toggleHud: () => void;
  toggleLeftDrawer: () => void;
  toggleRightDrawer: () => void;
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
}

export const useTwinStore = create<TwinState>((set, get) => ({
  currentFrame: null,
  history: [],
  cameraMode: 'meso',
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

  setFrame: (frame) =>
    set((state) => {
      const newHistory = [...state.history, frame].slice(-60); // keep last 60 frames (6s at 10Hz)
      return {
        currentFrame: frame,
        history: newHistory,
        // Trust the backend as the source of truth for which actions are live,
        // so the UI cannot drift out of sync with the simulation.
        activeInterventions: frame.active_interventions ?? state.activeInterventions,
      };
    }),

  setCameraMode: (mode) => set({ cameraMode: mode }),

  setSelectedSubsystem: (subsystem) => {
    set({ selectedSubsystem: subsystem });
  },

  setActiveInspection: (inspection) => {
    if (inspection) {
      set({ activeInspection: inspection, selectedSubsystem: inspection });
    } else {
      set({ activeInspection: null });
    }
  },

  toggleXray: () => set((state) => ({ xrayMode: !state.xrayMode })),

  toggleHud: () => set((state) => ({ isHudVisible: !state.isHudVisible })),
  toggleLeftDrawer: () => set((state) => ({ isLeftDrawerOpen: !state.isLeftDrawerOpen })),
  toggleRightDrawer: () => set((state) => ({ isRightDrawerOpen: !state.isRightDrawerOpen })),
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
        body: JSON.stringify({ force }),
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
    set({ aiAsking: true });
    try {
      const res = await fetch('/api/ai/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q }),
      });
      const data = await res.json();
      set((state) => ({
        aiAnswers: [...state.aiAnswers, { ...data, question: q }].slice(-8),
      }));
    } catch (err) {
      console.error('[AI] ask request failed:', err);
      set((state) => ({
        aiAnswers: [
          ...state.aiAnswers,
          {
            question: q,
            answer: 'Could not reach the assistant. Check that the backend is running.',
            source: 'error',
          },
        ].slice(-8),
      }));
    } finally {
      set({ aiAsking: false });
    }
  },

  initWebSocket: () => {
    const existing = get().ws;
    if (existing && (existing.readyState === WebSocket.OPEN || existing.readyState === WebSocket.CONNECTING)) {
      return;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.hostname || 'localhost';
    const wsUrl = `${protocol}//${host}:8000/ws/telemetry`;

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
}));
