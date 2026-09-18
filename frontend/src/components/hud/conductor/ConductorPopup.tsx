import React, { useEffect, useRef } from 'react';
import { HardHat, Maximize2, X } from 'lucide-react';
import { useTwinStore } from '../../../store/useTwinStore';
import { ConductorChat } from './ConductorChat';

/**
 * 360x480, anchored above the dock. Collapsing "What's monitored" to a chip
 * strip (handled in SubsystemSelector) and nudging the camera right (handled
 * in TwinCanvas) both key off conductorState === 'popup', set here.
 */
export const ConductorPopup: React.FC = () => {
  const setConductorState = useTwinStore((s) => s.setConductorState);
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setConductorState('docked');
    };
    document.addEventListener('keydown', onEsc);
    return () => document.removeEventListener('keydown', onEsc);
  }, [setConductorState]);

  return (
    <div
      ref={panelRef}
      className="absolute bottom-36 left-4 z-30 w-[360px] h-[480px] glass-panel-floating rounded pointer-events-auto flex flex-col overflow-hidden"
    >
      <div className="flex items-center justify-between px-2.5 py-2 border-b border-slate-200 shrink-0">
        <div className="flex items-center gap-2 min-w-0">
          <span className="w-6 h-6 rounded-full bg-ink-900 flex items-center justify-center shrink-0">
            <HardHat className="w-3 h-3 text-white" />
          </span>
          <div className="min-w-0">
            <div className="text-xs font-bold text-slate-900 leading-tight">Conductor</div>
            <div className="text-label text-slate-500 leading-tight truncate">
              Car 3 · reading live data
            </div>
          </div>
        </div>
        <div className="flex items-center gap-0.5 shrink-0">
          <button
            onClick={() => setConductorState('expanded')}
            className="p-1 rounded text-slate-400 hover:text-slate-800 hover:bg-slate-100 transition-colors"
            title="Expand"
          >
            <Maximize2 className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => setConductorState('docked')}
            className="p-1 rounded text-slate-400 hover:text-slate-800 hover:bg-slate-100 transition-colors"
            title="Close"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      <ConductorChat compact />
    </div>
  );
};
