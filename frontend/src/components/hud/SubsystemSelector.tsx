import React from 'react';
import {
  Activity,
  ChevronLeft,
  ChevronRight,
  DoorOpen,
  GitCommit,
  Layers,
  Wind,
} from 'lucide-react';
import { useTwinStore } from '../../store/useTwinStore';
import { InterventionAction, MetricStatus, SubsystemSelection } from '../../types/telemetry';
import { STATUS_SHORT, STATUS_STYLES, classifyMetric, plainTerm } from '../../lib/metricGlossary';

const VERDICT_RANK: Record<MetricStatus, number> = {
  ACTION_NEEDED: 3,
  WATCH: 2,
  GOOD: 1,
  UNKNOWN: 0,
};

/** Which repair action fixes which subsystem's single canonical row. */
const REPAIR_FOR_SUBSYSTEM: Partial<Record<SubsystemSelection, { action: InterventionAction; label: string }>> = {
  acv: { action: 'ACTION_REPLACE_FILTER', label: 'Replace the aircon filter' },
  door: { action: 'ACTION_LUBRICATE_DOOR', label: 'Grease the door tracks' },
  shm: { action: 'ACTION_INSPECT_BEARING', label: 'Service the wheel bearing' },
};

export const SubsystemSelector: React.FC = () => {
  const selectedSubsystem = useTwinStore((state) => state.selectedSubsystem);
  const setSelectedSubsystem = useTwinStore((state) => state.setSelectedSubsystem);
  const setRightDrawerTab = useTwinStore((state) => state.setRightDrawerTab);
  const currentFrame = useTwinStore((state) => state.currentFrame);
  const isLeftDrawerOpen = useTwinStore((state) => state.isLeftDrawerOpen);
  const toggleLeftDrawer = useTwinStore((state) => state.toggleLeftDrawer);
  const beginner = useTwinStore((state) => state.uiMode) === 'beginner';

  const subsystems = currentFrame?.subsystems;
  const status = currentFrame?.plain_status ?? {};

  interface Item {
    id: SubsystemSelection;
    plainLabel: string;
    technicalLabel: string;
    plainSub: string;
    technicalSub: string;
    icon: React.FC<{ className?: string }>;
    verdict?: MetricStatus;
  }

  const items: Item[] = [
    {
      id: 'overview',
      plainLabel: 'Whole train',
      technicalLabel: 'SMRT Fleet Matrix',
      plainSub: 'Everything at a glance',
      technicalSub: 'Train-Track Infrastructure',
      icon: Layers,
    },
    {
      id: 'door',
      plainLabel: 'Passenger doors',
      technicalLabel: 'Door 3R System',
      plainSub: subsystems
        ? `${plainTerm(subsystems.door.cycle_state)} • ${subsystems.door.motor_current_amps.toFixed(1)} A`
        : 'Doors and their motors',
      technicalSub: subsystems
        ? `${subsystems.door.cycle_state} • ${subsystems.door.motor_current_amps}A`
        : 'Electromechanics',
      icon: DoorOpen,
      verdict:
        status['door.anomaly_score'] ??
        classifyMetric('door.anomaly_score', subsystems?.door.anomaly_score),
    },
    {
      id: 'acv',
      plainLabel: 'Air-conditioning',
      technicalLabel: 'ACV Climate Pack',
      plainSub: subsystems
        ? `cools by ${subsystems.acv.delta_temp_c.toFixed(1)}°C`
        : 'Cabin cooling',
      technicalSub: subsystems
        ? `ΔT ${subsystems.acv.delta_temp_c}°C • ${subsystems.acv.compressor_power_kw}kW`
        : 'Thermodynamics',
      icon: Wind,
      verdict:
        status['acv.efficiency_rating'] ??
        classifyMetric('acv.efficiency_rating', subsystems?.acv.efficiency_rating),
    },
    {
      id: 'shm',
      plainLabel: 'Wheels & frame',
      technicalLabel: 'Bogie SHM',
      plainSub: subsystems
        ? `shaking ${subsystems.shm.vibration_rms_g.toFixed(2)} g`
        : 'Wheel assembly condition',
      technicalSub: subsystems
        ? `${subsystems.shm.vibration_rms_g}g RMS • ${subsystems.shm.peak_frequency_hz}Hz`
        : 'Fatigue Dynamics',
      icon: Activity,
      verdict:
        status['shm.vibration_rms_g'] ??
        classifyMetric('shm.vibration_rms_g', subsystems?.shm.vibration_rms_g),
    },
    {
      id: 'rail',
      plainLabel: 'Track condition',
      technicalLabel: 'LTA Rail Infrastructure',
      plainSub: subsystems
        ? `ripples ${subsystems.rail_corrugation.depth_microns.toFixed(0)} μm deep`
        : 'Rail surface wear',
      technicalSub: subsystems
        ? `${subsystems.rail_corrugation.depth_microns}μm • ${subsystems.rail_corrugation.wavelength_class}`
        : 'Track Corrugation',
      icon: GitCommit,
      verdict:
        status['rail_corrugation.severity_score'] ??
        classifyMetric(
          'rail_corrugation.severity_score',
          subsystems?.rail_corrugation.severity_score
        ),
    },
  ];

  // Single alarm system: the worst subsystem is promoted to the top and gets
  // the danger spine + action link; "Whole train" is a nav item, not a status
  // row, so it always sits last.
  const overviewItem = items.find((i) => i.id === 'overview')!;
  const subsystemItems = items
    .filter((i) => i.id !== 'overview')
    .sort((a, b) => VERDICT_RANK[b.verdict ?? 'UNKNOWN'] - VERDICT_RANK[a.verdict ?? 'UNKNOWN']);
  const orderedItems = [...subsystemItems, overviewItem];
  const topVerdict = subsystemItems[0]?.verdict ?? 'UNKNOWN';

  return (
    <div
      className={`absolute top-28 left-4 z-20 transition-colors duration-150 duration-300 pointer-events-none ${
        isLeftDrawerOpen ? 'w-[286px]' : 'w-12'
      }`}
    >
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
          const isPromoted = idx === 0 && VERDICT_RANK[topVerdict] > VERDICT_RANK.GOOD;
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
              {isPromoted && (
                <div className="absolute left-0 top-0 bottom-0 w-1 rounded-l bg-status-fault" aria-hidden="true" />
              )}
              <button
                onClick={() => setSelectedSubsystem(item.id)}
                title={!isLeftDrawerOpen ? label : undefined}
                className={`w-full text-left transition-colors duration-150 flex items-center ${
                  isLeftDrawerOpen ? `p-2.5 ${isPromoted ? 'pl-3.5' : ''} justify-between` : 'p-2 justify-center'
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

              {isPromoted && isLeftDrawerOpen && repair && (
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
    </div>
  );
};