import React from 'react';
import { HardHat } from 'lucide-react';
import { useTwinStore } from '../../../store/useTwinStore';

/**
 * Docked default state: furniture, not a floating attention-grabbing pet.
 * Sits at the bottom of the left rail, above the timeline scrubber.
 */
export const ConductorDock: React.FC = () => {
  const setConductorState = useTwinStore((s) => s.setConductorState);

  return (
    <div className="absolute bottom-24 left-4 z-20 pointer-events-none">
      <button
        onClick={() => setConductorState('popup')}
        className="glass-panel pointer-events-auto rounded-full pl-1 pr-3 py-1 flex items-center gap-2 hover:border-ink-500 transition-colors duration-150"
      >
        <span className="relative w-7 h-7 rounded-full bg-ink-900 flex items-center justify-center shrink-0">
          <HardHat className="w-3.5 h-3.5 text-white" />
          <span className="absolute -bottom-0.5 -right-0.5 w-2 h-2 rounded-full bg-status-nominal ring-2 ring-white" />
        </span>
        <span className="text-xs font-semibold text-slate-700">Ask the Conductor</span>
        <span className="text-label font-mono text-slate-400">⌘K</span>
      </button>
    </div>
  );
};
