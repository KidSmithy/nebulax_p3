import React, { useEffect } from 'react';
import { useTwinStore } from './store/useTwinStore';
import { TwinCanvas } from './components/canvas/TwinCanvas';
import { CockpitHeader } from './components/hud/CockpitHeader';
import { ZoomSlider } from './components/hud/ZoomSlider';
import { ResolvedLogPanel } from './components/hud/ResolvedLogPanel';

import { ConductorDock } from './components/hud/conductor/ConductorDock';
import { ConductorPopup } from './components/hud/conductor/ConductorPopup';
import { ConductorExpanded } from './components/hud/conductor/ConductorExpanded';
import { ConductorOnboarding } from './components/hud/conductor/ConductorOnboarding';

export const App: React.FC = () => {
  const initWebSocket = useTwinStore((state) => state.initWebSocket);
  const disconnectWebSocket = useTwinStore((state) => state.disconnectWebSocket);
  const resetToSeededFindings = useTwinStore((state) => state.resetToSeededFindings);
  const conductorEnabled = useTwinStore((state) => state.conductorEnabled);
  const conductorState = useTwinStore((state) => state.conductorState);
  const setConductorState = useTwinStore((state) => state.setConductorState);

  useEffect(() => {
    // Findings are backend state, so a refresh would otherwise inherit whatever
    // the last session left behind. Re-assert the intended start on every load -
    // four inspection tags on NSL, none on EWL - then stream. The socket is
    // opened without waiting, since frames carry the findings and simply show
    // the re-seeded ones as soon as the call lands.
    void resetToSeededFindings();
    initWebSocket();
    return () => disconnectWebSocket();
  }, [initWebSocket, disconnectWebSocket, resetToSeededFindings]);

  // Global ⌘K / Ctrl+K, matching the hint shown on the docked Conductor row.
  useEffect(() => {
    if (!conductorEnabled) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setConductorState(conductorState === 'docked' ? 'popup' : 'docked');
      }
    };
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [conductorEnabled, conductorState, setConductorState]);

  return (
    <div className="w-screen h-screen relative bg-slate-100 overflow-hidden font-sans select-none xl:flex">
      {conductorEnabled && conductorState === 'onboarding' && <ConductorOnboarding />}
      {conductorEnabled && conductorState === 'expanded' && <ConductorExpanded />}

      {/* Everything below reflows into the remaining width when the
          Conductor is expanded at >=1280px, and fills the screen otherwise. */}
      <div className="relative flex-1 min-w-0 h-full">
        {/* 3D Scene Viewport */}
        <div className="absolute inset-0 z-0">
          <TwinCanvas />
        </div>

        {/* Cockpit HUD Overlays */}
        <CockpitHeader />
        <ZoomSlider />
        <ResolvedLogPanel />

        {conductorEnabled && conductorState === 'docked' && <ConductorDock />}
        {conductorEnabled && conductorState === 'popup' && <ConductorPopup />}
      </div>
    </div>
  );
};

export default App;
