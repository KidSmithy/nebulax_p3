import React from 'react';
import { Train, Wrench } from 'lucide-react';
import { useTwinStore } from '../../store/useTwinStore';

export const CockpitHeader: React.FC = () => {
  const activeInterventions = useTwinStore((state) => state.activeInterventions);
  const repairCount = Object.values(activeInterventions).filter(Boolean).length;

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
      {repairCount > 0 && (
        <div className="flex items-center gap-2 pointer-events-auto min-w-0 flex-1 flex-wrap justify-end">
          <div className="glass-panel border border-emerald-300 bg-emerald-50/80 px-2.5 py-1.5 rounded flex items-center space-x-1.5">
            <Wrench className="w-3.5 h-3.5 text-emerald-700" />
            <div>
              <div className="text-[8px] font-mono text-emerald-700 opacity-80">
                Simulated
              </div>
              <div className="text-label font-mono font-bold leading-none text-emerald-800">
                {repairCount} repair{repairCount > 1 ? 's' : ''}
              </div>
            </div>
          </div>
        </div>
      )}
    </header>
  );
};