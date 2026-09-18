import React, { useEffect } from 'react';
import { useTwinStore } from './store/useTwinStore';
import { TwinCanvas } from './components/canvas/TwinCanvas';
import { CockpitHeader } from './components/hud/CockpitHeader';
import { SubsystemSelector } from './components/hud/SubsystemSelector';
import { RightDrawer } from './components/hud/RightDrawer';
import { TimelineScrubber } from './components/hud/TimelineScrubber';

export const App: React.FC = () => {
  const initWebSocket = useTwinStore((state) => state.initWebSocket);
  const disconnectWebSocket = useTwinStore((state) => state.disconnectWebSocket);

  useEffect(() => {
    initWebSocket();
    return () => disconnectWebSocket();
  }, [initWebSocket, disconnectWebSocket]);

  const isHudVisible = useTwinStore((state) => state.isHudVisible);

  return (
    <div className="w-screen h-screen relative bg-slate-100 overflow-hidden font-sans select-none">
      {/* 3D Scene Viewport */}
      <div className="absolute inset-0 z-0">
        <TwinCanvas />
      </div>

      {/* Cockpit HUD Overlays */}
      <CockpitHeader />
      {isHudVisible && (
        <>
          <SubsystemSelector />
          <RightDrawer />
          <TimelineScrubber />
        </>
      )}
    </div>
  );

};

export default App;
