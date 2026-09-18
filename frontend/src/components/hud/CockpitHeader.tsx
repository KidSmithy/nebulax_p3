import React from 'react';
import { Train, Wrench } from 'lucide-react';
import { useTwinStore } from '../../store/useTwinStore';
import { CameraPreset } from '../../types/telemetry';

const CAMERA_COPY: Record<CameraPreset, { plain: string; tip: string }> = {
  macro: { plain: 'All 8 cars', tip: 'Pull back to see every car and which one needs attention.' },
  meso: { plain: 'One car', tip: 'Look at Car 3 as a whole.' },
  micro: { plain: 'Close-up', tip: 'Zoom in on the selected component.' },
};


export const CockpitHeader: React.FC = () => {
  const activeInterventions = useTwinStore((state) => state.activeInterventions);
  const repairCount = Object.values(activeInterventions).filter(Boolean).length;
  const cameraMode = useTwinStore((state) => state.cameraMode);
  const setCameraMode = useTwinStore((state) => state.setCameraMode);

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
              <span className="text-label bg-slate-100 text-ink-700 font-mono px-1 py-0.5 rounded border border-red-200 font-bold whitespace-nowrap">
                NSL
              </span>
            </div>
            <p className="text-label text-slate-500 font-mono whitespace-nowrap">
              C151B-SET-402 • CAR 3 • NORTH-SOUTH LINE
            </p>
          </div>
        </div>
      </div>

      {/* Right controls */}
      <div className="flex items-center gap-2 pointer-events-auto min-w-0 flex-1 flex-wrap justify-end">
        {repairCount > 0 && (
          <div className="glass-panel border border-emerald-300 bg-emerald-50/80 px-2.5 py-1.5 rounded flex items-center space-x-1.5">
            <Wrench className="w-3.5 h-3.5 text-emerald-700" />
            <div>
              <div className="text-[8px] font-mono text-emerald-700 opacity-80">Simulated</div>
              <div className="text-label font-mono font-bold leading-none text-emerald-800">
                {repairCount} repair{repairCount > 1 ? 's' : ''}
              </div>
            </div>
          </div>
        )}

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