import { create } from 'zustand';
import { UnifiedTelemetryFrame, CameraPreset, SubsystemSelection } from '../types/telemetry';

interface TwinState {
  currentFrame: UnifiedTelemetryFrame | null;
  history: UnifiedTelemetryFrame[];
  cameraMode: CameraPreset;
  selectedSubsystem: SubsystemSelection;
  xrayMode: boolean;
  isPlaying: boolean;
  playbackSpeed: number;
  isConnected: boolean;
  activeInterventions: {
    ACTION_GRIND_RAIL: boolean;
    ACTION_LUBRICATE_DOOR: boolean;
    ACTION_REPLACE_FILTER: boolean;
    ACTION_INSPECT_BEARING: boolean;
  };
  ws: WebSocket | null;

  // UI Decluttering & Layout Controls
  isHudVisible: boolean;
  isLeftDrawerOpen: boolean;
  isRightDrawerOpen: boolean;
  rightDrawerTab: 'telemetry' | 'whatif';
  activeInspection: 'door' | 'acv' | 'shm' | 'rail' | null;

  // Actions
  setFrame: (frame: UnifiedTelemetryFrame) => void;
  setCameraMode: (mode: CameraPreset) => void;
  setSelectedSubsystem: (subsystem: SubsystemSelection) => void;
  setActiveInspection: (inspection: 'door' | 'acv' | 'shm' | 'rail' | null) => void;
  toggleXray: () => void;
  toggleHud: () => void;
  toggleLeftDrawer: () => void;
  toggleRightDrawer: () => void;
  setRightDrawerTab: (tab: 'telemetry' | 'whatif') => void;
  setPlaying: (playing: boolean) => void;
  setPlaybackSpeed: (speed: number) => void;
  setIntervention: (action: string, enabled: boolean) => void;
  setConnected: (connected: boolean) => void;
  initWebSocket: () => void;
  sendSeek: (kp: number) => void;
  triggerWhatIf: (action: string, enabled: boolean) => void;
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
  activeInterventions: {
    ACTION_GRIND_RAIL: false,
    ACTION_LUBRICATE_DOOR: false,
    ACTION_REPLACE_FILTER: false,
    ACTION_INSPECT_BEARING: false,
  },
  ws: null,

  // UI state defaults (right drawer starts closed for clean spacious 3D view)
  isHudVisible: true,
  isLeftDrawerOpen: true,
  isRightDrawerOpen: false,
  rightDrawerTab: 'telemetry',
  activeInspection: null,


  setFrame: (frame) =>
    set((state) => {
      const newHistory = [...state.history, frame].slice(-60); // keep last 60 frames (6s at 10Hz)
      return { currentFrame: frame, history: newHistory };
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

  initWebSocket: () => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.hostname || 'localhost';
    const wsUrl = `${protocol}//${host}:8000/ws/telemetry`;

    try {
      const socket = new WebSocket(wsUrl);

      socket.onopen = () => {
        set({ isConnected: true, ws: socket });
        console.log('[WebSocket] Connected to digital twin telemetry stream');
      };

      socket.onmessage = (event) => {
        try {
          const frame: UnifiedTelemetryFrame = JSON.parse(event.data);
          get().setFrame(frame);
        } catch (err) {
          console.error('[WebSocket] Error parsing frame:', err);
        }
      };

      socket.onclose = () => {
        set({ isConnected: false, ws: null });
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
}));
