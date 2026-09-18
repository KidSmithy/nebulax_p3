import React from 'react';
import { Train } from 'lucide-react';
import { useTwinStore } from '../../store/useTwinStore';
import { CameraPreset } from '../../types/telemetry';
import { LINE_IDS, LINES } from '../../lib/lines';

const CAMERA_COPY: Record<CameraPreset, { plain: string; tip: string }> = {
  macro: { plain: 'All 8 cars', tip: 'Pull back to see every car and which one needs attention.' },
  meso: { plain: 'One car', tip: 'Look at the monitored car as a whole.' },
  micro: { plain: 'Close-up', tip: 'Zoom in on the selected component.' },
};


export const CockpitHeader: React.FC = () => {
  const cameraMode = useTwinStore((state) => state.cameraMode);
  const setCameraMode = useTwinStore((state) => state.setCameraMode);
  const monitoredCar = useTwinStore((state) => state.monitoredCar);
  const activeLine = useTwinStore((state) => state.activeLine);
  const setActiveLine = useTwinStore((state) => state.setActiveLine);
  const line = LINES[activeLine];

  return (
    <header className="absolute top-4 left-4 right-4 z-30 flex items-start pointer-events-none gap-2">
      {/* Left branding and fleet info */}
      <div className="flex items-center space-x-3 pointer-events-auto shrink-0">
        <div className="glass-panel-glow px-3.5 py-2 rounded flex items-center space-x-3 shrink-0 whitespace-nowrap">
          <div className="w-8 h-8 rounded bg-ink-900 border border-ink-700 flex items-center justify-center text-white shrink-0">
            <Train className="w-4 h-4 stroke-[2.5]" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-xs font-black text-slate-900 flex items-center space-x-1.5 whitespace-nowrap">
                <span className="text-ink-900 font-extrabold">SMRT</span>
                <span>•</span>
                <span>LTA Digital Twin</span>
              </h1>
              <span
                className="text-label bg-slate-100 text-ink-700 font-mono px-1 py-0.5 rounded border font-bold whitespace-nowrap"
                style={{ borderColor: line.color }}
              >
                {activeLine}
              </span>
            </div>
            <p className="text-label text-slate-500 font-mono whitespace-nowrap">
              {line.setId} • CAR {monitoredCar} • {line.longName}
            </p>
          </div>
        </div>
      </div>

      {/* Right controls */}
      <div className="flex items-center gap-2 pointer-events-auto min-w-0 flex-1 flex-wrap justify-end">
        {/* Line tabs: each line is its own train with its own findings and Conductor */}
        <div className="glass-panel p-1 rounded flex items-center space-x-1 text-xs font-mono">
          {LINE_IDS.map((id) => (
            <button
              key={id}
              onClick={() => setActiveLine(id)}
              className={`px-2 py-1 rounded text-label transition-colors duration-150 font-bold flex items-center gap-1.5 ${
                activeLine === id
                  ? 'bg-ink-900 text-white'
                  : 'text-slate-500 hover:text-slate-800 hover:bg-slate-100'
              }`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${LINES[id].dotClass}`} aria-hidden="true" />
              {LINES[id].tab}
            </button>
          ))}
        </div>

        {/* Camera presets */}
        <div className="glass-panel p-1 rounded flex items-center space-x-1 text-xs font-mono">
          {(['macro', 'meso', 'micro'] as CameraPreset[]).map((preset) => (
            <button
              key={preset}
              onClick={() => setCameraMode(preset)}
              title={CAMERA_COPY[preset].tip}
              className={`px-2 py-1 rounded text-label transition-colors duration-150 font-bold ${
                cameraMode === preset
                  ? 'bg-ink-900 text-white'
                  : 'text-slate-500 hover:text-slate-800 hover:bg-slate-100'
              }`}
            >
              {CAMERA_COPY[preset].plain}
            </button>
          ))}
        </div>
      </div>
    </header>
  );
};