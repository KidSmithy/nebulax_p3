import React from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { useTwinStore } from '../../store/useTwinStore';
import { STATUS_SHORT, STATUS_STYLES } from '../../lib/metricGlossary';
import { REPAIR_FOR_SUBSYSTEM, useMonitoredItems } from '../../lib/useMonitoredItems';

/** Collapsed summary strip shown while the Conductor popup is open, so the
 *  popup gets clean vertical space instead of competing with the full rail. */
const MonitoredChipStrip: React.FC = () => {
  const { orderedItems, isPromoted } = useMonitoredItems();
  const beginner = useTwinStore((state) => state.uiMode) === 'beginner';
  const rows = orderedItems.filter((i) => i.id !== 'overview');
  const hiddenCount = Math.max(0, rows.length - 3);

  return (
    <div className="glass-panel p-2 rounded pointer-events-auto flex flex-wrap items-center gap-1.5">
      {rows.slice(0, 3).map((item, idx) => {
        const verdict = item.verdict ?? 'UNKNOWN';
        const styles = STATUS_STYLES[verdict];
        const label = beginner ? item.plainLabel : item.technicalLabel;
        return (
          <span
            key={item.id}
            className={`text-label font-semibold px-2 py-1 rounded-full border ${styles.text} ${styles.border} bg-white ${
              isPromoted(item.id, idx) ? 'border-2' : ''
            }`}
          >
            {label.split(' ')[0]} {STATUS_SHORT[verdict]}
          </span>
        );
      })}
      {hiddenCount > 0 && (
        <span className="text-label text-slate-500">+{hiddenCount} watching</span>
      )}
    </div>
  );
};

export const SubsystemSelector: React.FC = () => {
  const selectedSubsystem = useTwinStore((state) => state.selectedSubsystem);
  const setSelectedSubsystem = useTwinStore((state) => state.setSelectedSubsystem);
  const setRightDrawerTab = useTwinStore((state) => state.setRightDrawerTab);
  const isLeftDrawerOpen = useTwinStore((state) => state.isLeftDrawerOpen);
  const toggleLeftDrawer = useTwinStore((state) => state.toggleLeftDrawer);
  const beginner = useTwinStore((state) => state.uiMode) === 'beginner';
  const conductorState = useTwinStore((state) => state.conductorState);

  const { orderedItems, isPromoted } = useMonitoredItems();

  // The expanded Conductor panel docks its own pinned compact copy of this
  // list at its top, so the full rail steps aside entirely rather than
  // showing twice.
  if (conductorState === 'expanded') return null;

  return (
    <div
      className={`absolute top-28 left-4 z-20 transition-colors duration-150 duration-300 pointer-events-none ${
        conductorState === 'popup' ? 'w-auto' : isLeftDrawerOpen ? 'w-[286px]' : 'w-12'
      }`}
    >
      {conductorState === 'popup' ? (
        <MonitoredChipStrip />
      ) : (
        <div className="glass-panel p-2 rounded pointer-events-auto flex flex-col space-y-1.5">
          <div
            className={`flex items-center justify-between px-1 py-1 border-b border-slate-200 mb-1 ${
              !isLeftDrawerOpen && 'justify-center'
            }`}
          >
            {isLeftDrawerOpen && (
              <span className="text-label font-bold text-slate-500">
                {beginner ? "What's monitored" : 'SMRT Subsystems'}
              </span>
            )}
            <button
              onClick={toggleLeftDrawer}
              className="p-1 rounded text-slate-400 hover:text-slate-800 hover:bg-slate-100 transition-colors"
              title={isLeftDrawerOpen ? 'Collapse panel' : 'Expand panel'}
            >
              {isLeftDrawerOpen ? (
                <ChevronLeft className="w-3.5 h-3.5" />
              ) : (
                <ChevronRight className="w-3.5 h-3.5" />
              )}
            </button>
          </div>

          {orderedItems.map((item, idx) => {
            const isSelected = selectedSubsystem === item.id;
            const Icon = item.icon;
            const verdict = item.verdict ?? 'UNKNOWN';
            const styles = STATUS_STYLES[verdict];
            const label = beginner ? item.plainLabel : item.technicalLabel;
            const sub = beginner ? item.plainSub : item.technicalSub;
            const promoted = isPromoted(item.id, idx);
            const repair = REPAIR_FOR_SUBSYSTEM[item.id];

            return (
              <div
                key={item.id}
                className={`relative rounded border overflow-hidden ${
                  isSelected
                    ? 'border-ink-200 bg-slate-100'
                    : 'border-slate-200 hover:border-slate-300 hover:bg-slate-50'
                }`}
              >
                {promoted && (
                  <div className="absolute left-0 top-0 bottom-0 w-1 rounded-l bg-status-fault" aria-hidden="true" />
                )}
                <button
                  onClick={() => setSelectedSubsystem(item.id)}
                  title={!isLeftDrawerOpen ? label : undefined}
                  className={`w-full text-left transition-colors duration-150 flex items-center ${
                    isLeftDrawerOpen ? `p-2.5 ${promoted ? 'pl-3.5' : ''} justify-between` : 'p-2 justify-center'
                  } ${isSelected ? 'text-ink-900' : 'text-slate-700'}`}
                >
                  <div className="flex items-center space-x-2.5 min-w-0">
                    <div
                      className={`w-7 h-7 rounded flex items-center justify-center transition-colors shrink-0 ${
                        isSelected
                          ? 'bg-ink-900 text-white'
                          : item.verdict
                          ? `${styles.bg} ${styles.text} border ${styles.border}`
                          : 'bg-slate-100 text-slate-500'
                      }`}
                    >
                      <Icon className="w-4 h-4" />
                    </div>
                    {isLeftDrawerOpen && (
                      <div className="min-w-0">
                        <div className="text-xs font-semibold text-slate-800">{label}</div>
                        <div className="text-label text-slate-500 font-mono leading-snug">
                          {sub}
                        </div>
                      </div>
                    )}
                  </div>

                  {isLeftDrawerOpen && item.verdict && (
                    <div
                      className={`text-label font-bold px-1.5 py-0.5 rounded-full border shrink-0 ${styles.text} ${styles.border} bg-white`}
                    >
                      {STATUS_SHORT[verdict]}
                    </div>
                  )}
                </button>

                {promoted && isLeftDrawerOpen && repair && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setSelectedSubsystem(item.id);
                      setRightDrawerTab('whatif');
                      useTwinStore.setState({ isRightDrawerOpen: true });
                    }}
                    className="w-full text-left px-2.5 pb-2 pl-3.5 pt-1 border-t border-slate-200 text-label font-semibold text-ink-700 hover:text-ink-900 transition-colors duration-150"
                  >
                    Fix: {repair.label} →
                  </button>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
