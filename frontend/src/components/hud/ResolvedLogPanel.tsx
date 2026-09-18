import React, { useEffect, useRef, useState } from 'react';
import { Activity, ChevronDown, ChevronUp, DoorOpen, GitCommit, ReceiptText, Wind } from 'lucide-react';
import { useTwinStore } from '../../store/useTwinStore';
import { STATUS_STYLES } from '../../lib/metricGlossary';
import { ACTIONS } from './WhatIfPanel';
import { LINES } from '../../lib/lines';

const SUBSYSTEM_ICON: Record<string, React.FC<{ className?: string }>> = {
  door: DoorOpen,
  acv: Wind,
  shm: Activity,
  rail: GitCommit,
};

const SUBSYSTEM_LABEL: Record<string, string> = {
  door: 'Passenger doors',
  acv: 'Air-conditioning',
  shm: 'Wheels & frame',
  rail: 'Track condition',
};

const SGT_FORMAT = new Intl.DateTimeFormat('en-GB', {
  timeZone: 'Asia/Singapore',
  day: '2-digit',
  month: '2-digit',
  hour: '2-digit',
  minute: '2-digit',
  hourCycle: 'h23',
});

/** Backend timestamps are UTC ISO strings; show them as e.g. "19/09 01:53" in Singapore time. */
function formatSgt(iso?: string): string {
  const d = iso ? new Date(iso) : null;
  if (!d || Number.isNaN(d.getTime())) return '--';
  const p = Object.fromEntries(SGT_FORMAT.formatToParts(d).map((x) => [x.type, x.value]));
  return `${p.day}/${p.month} ${p.hour}:${p.minute}`;
}

/**
 * Bottom-right, continuously appended receipt of every resolved finding -
 * persists across reloads because it's read straight from the backend's
 * in-memory log (GET /api/log), not local/session state.
 */
export const ResolvedLogPanel: React.FC = () => {
  const log = useTwinStore((s) => s.resolvedLog);
  const fetchLog = useTwinStore((s) => s.fetchLog);
  const [open, setOpen] = useState(true);
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchLog();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    // Keep the newest entry in view without scrolling the page itself.
    if (open) listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: 'smooth' });
  }, [log.length, open]);

  return (
    <div className="absolute bottom-24 right-4 z-20 w-80 pointer-events-none">
      <div className="glass-panel rounded pointer-events-auto flex flex-col overflow-hidden">
        <button
          onClick={() => setOpen((o) => !o)}
          className="w-full flex items-center justify-between px-2.5 py-1.5 border-b border-slate-200 shrink-0"
        >
          <span className="flex items-center gap-1.5 text-label font-bold text-slate-500">
            <ReceiptText className="w-3.5 h-3.5" />
            RESOLVED LOG
            {log.length > 0 && <span className="text-slate-400 font-mono">({log.length})</span>}
          </span>
          {open ? <ChevronDown className="w-3.5 h-3.5 text-slate-400" /> : <ChevronUp className="w-3.5 h-3.5 text-slate-400" />}
        </button>

        {open && (
          /* Rows are a fixed 54px, so 270px shows exactly 5; anything older scrolls. */
          <div ref={listRef} className="overflow-y-auto custom-scrollbar-lg max-h-[270px]">
            {log.length === 0 ? (
              <p className="px-2.5 py-3 text-label text-slate-400 leading-relaxed">
                Nothing resolved yet. Findings you resolve from the 3D view append here.
              </p>
            ) : (
              log.map((entry, i) => {
                const Icon = SUBSYSTEM_ICON[entry.subsystem] ?? Activity;
                const styles = STATUS_STYLES[entry.status] ?? STATUS_STYLES.UNKNOWN;
                const repair = ACTIONS.find((a) => a.action === entry.recommended_action);
                const time = formatSgt(entry.resolved_at_timestamp);
                return (
                  <div key={entry.id} className="flex gap-2 px-2.5 py-2 min-h-[54px]">
                    {/* Timeline rail: a dot per entry, connected by a vertical line. */}
                    <div className="flex flex-col items-center shrink-0 pt-0.5">
                      <span className={`w-2 h-2 rounded-full ${styles.bar}`} />
                      {i < log.length - 1 && <span className="w-px flex-1 bg-slate-200 mt-1" />}
                    </div>

                    <div className="min-w-0 flex-1 pb-1">
                      <div className="flex items-center justify-between gap-2">
                        <div className="flex items-center gap-1.5 min-w-0">
                          <Icon className={`w-3 h-3 shrink-0 ${styles.text}`} />
                          <span className="text-[11px] font-semibold text-slate-800 truncate">
                            {SUBSYSTEM_LABEL[entry.subsystem] ?? entry.subsystem}
                          </span>
                          <span className="text-label font-mono shrink-0 text-slate-400">
                            <span className={`font-bold ${LINES[entry.line ?? 'NSL'].textClass}`}>
                              {entry.line ?? 'NSL'}
                            </span>
                            {entry.car != null && ` · Car ${entry.car}`}
                          </span>
                        </div>
                        <span className="text-label font-mono text-slate-400 shrink-0">{time}</span>
                      </div>
                      {repair && (
                        <div className="text-[10.5px] text-slate-600 mt-0.5">→ {repair.plainTitle}</div>
                      )}
                    </div>
                  </div>
                );
              })
            )}
          </div>
        )}
      </div>
    </div>
  );
};
