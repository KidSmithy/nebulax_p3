import React, { useState } from 'react';
import { ChevronDown, ChevronUp, HardHat, Minimize2, X } from 'lucide-react';
import { useTwinStore } from '../../../store/useTwinStore';
import { STATUS_SHORT, STATUS_STYLES } from '../../../lib/metricGlossary';
import { useMonitoredItems } from '../../../lib/useMonitoredItems';
import { ConductorChat } from './ConductorChat';

/**
 * 420px full-height left panel. "What's monitored" is pinned at the top in
 * compact single-line form - not replaced - so the fault stays visible
 * directly above the sentence explaining it.
 */
export const ConductorExpanded: React.FC = () => {
  const setConductorState = useTwinStore((s) => s.setConductorState);
  const monitoredCar = useTwinStore((s) => s.monitoredCar);
  const beginner = useTwinStore((s) => s.uiMode) === 'beginner';
  const { orderedItems, isPromoted } = useMonitoredItems();
  const [monitoredOpen, setMonitoredOpen] = useState(true);

  const rows = orderedItems.filter((i) => i.id !== 'overview');

  return (
    <div className="fixed xl:static top-0 left-0 h-full w-[420px] z-40 xl:z-auto bg-white border-r border-slate-200 flex flex-col shrink-0">
      <div className="flex items-center justify-between px-3 py-2.5 border-b border-slate-200 shrink-0">
        <div className="flex items-center gap-2 min-w-0">
          <span className="w-7 h-7 rounded-full bg-ink-900 flex items-center justify-center shrink-0">
            <HardHat className="w-3.5 h-3.5 text-white" />
          </span>
          <div className="min-w-0">
            <div className="text-sm font-bold text-slate-900 leading-tight">Conductor</div>
            <div className="text-label text-slate-500 leading-tight truncate">
              Car {monitoredCar} · reading live data
            </div>
          </div>
        </div>
        <div className="flex items-center gap-0.5 shrink-0">
          <button
            onClick={() => setConductorState('popup')}
            className="p-1 rounded text-slate-400 hover:text-slate-800 hover:bg-slate-100 transition-colors"
            title="Collapse to popup"
          >
            <Minimize2 className="w-4 h-4" />
          </button>
          <button
            onClick={() => setConductorState('docked')}
            className="p-1 rounded text-slate-400 hover:text-slate-800 hover:bg-slate-100 transition-colors"
            title="Close"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className="border-b border-slate-200 shrink-0">
        <button
          onClick={() => setMonitoredOpen((o) => !o)}
          className="w-full flex items-center justify-between px-3 py-1.5 text-label font-bold text-slate-500"
        >
          <span>{beginner ? "WHAT'S MONITORED" : 'SMRT SUBSYSTEMS'}</span>
          {monitoredOpen ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
        </button>
        {monitoredOpen && (
          <div className="px-3 pb-2 space-y-1">
            {rows.map((item, idx) => {
              const verdict = item.verdict ?? 'UNKNOWN';
              const styles = STATUS_STYLES[verdict];
              const label = beginner ? item.plainLabel : item.technicalLabel;
              const promoted = isPromoted(item.id, idx);
              return (
                <div
                  key={item.id}
                  className={`relative flex items-center justify-between rounded pl-2 pr-1.5 py-1 overflow-hidden ${
                    promoted ? 'bg-slate-50' : ''
                  }`}
                >
                  {promoted && (
                    <div className="absolute left-0 top-0 bottom-0 w-1 bg-status-fault" aria-hidden="true" />
                  )}
                  <span className={`text-xs font-medium ${promoted ? 'pl-1.5' : ''} text-slate-700 truncate`}>
                    {label}
                  </span>
                  {item.verdict && (
                    <span
                      className={`text-label font-bold px-1.5 py-0.5 rounded-full border shrink-0 ${styles.text} ${styles.border} bg-white`}
                    >
                      {STATUS_SHORT[verdict]}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>

      <div className="flex-1 min-h-0">
        <ConductorChat compact={false} />
      </div>
    </div>
  );
};
