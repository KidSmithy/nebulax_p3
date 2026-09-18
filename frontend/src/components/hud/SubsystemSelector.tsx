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
import { MetricStatus, SubsystemSelection } from '../../types/telemetry';
import { STATUS_SHORT, STATUS_STYLES, classifyMetric, plainTerm } from '../../lib/metricGlossary';

export const SubsystemSelector: React.FC = () => {
  const selectedSubsystem = useTwinStore((state) => state.selectedSubsystem);
  const setSelectedSubsystem = useTwinStore((state) => state.setSelectedSubsystem);
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

  return (
    <div
      className={`absolute top-20 left-4 z-20 transition-colors duration-150 duration-300 pointer-events-none ${
        isLeftDrawerOpen ? 'w-64' : 'w-12'
      }`}
    >
      <div className="glass-panel p-2 rounded pointer-events-auto flex flex-col space-y-1.5 shadow-2xl">
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

        {items.map((item) => {
          const isSelected = selectedSubsystem === item.id;
          const Icon = item.icon;
          const verdict = item.verdict ?? 'UNKNOWN';
          const styles = STATUS_STYLES[verdict];
          const label = beginner ? item.plainLabel : item.technicalLabel;
          const sub = beginner ? item.plainSub : item.technicalSub;

          return (
            <button
              key={item.id}
              onClick={() => setSelectedSubsystem(item.id)}
              title={!isLeftDrawerOpen ? label : undefined}
              className={`text-left transition-colors duration-150 rounded border flex items-center ${
                isLeftDrawerOpen ? 'p-2.5 justify-between' : 'p-2 justify-center'
              } ${
                isSelected
                  ? 'border-ink-200 bg-slate-100 text-ink-900'
                  : 'border-slate-200 hover:border-slate-300 hover:bg-slate-50 text-slate-700'
              }`}
            >
              <div className="flex items-center space-x-2.5 min-w-0">
                <div
                  className={`w-7 h-7 rounded flex items-center justify-center transition-colors shrink-0 ${
                    isSelected
                      ? 'bg-ink-900 text-white'
                      : item.verdict
                      ? `${styles.bg} ${styles.text}`
                      : 'bg-slate-100 text-slate-500'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                </div>
                {isLeftDrawerOpen && (
                  <div className="min-w-0">
                    <div className="text-xs font-semibold text-slate-800 truncate">{label}</div>
                    <div className="text-label text-slate-500 font-mono truncate max-w-[130px]">
                      {sub}
                    </div>
                  </div>
                )}
              </div>

              {isLeftDrawerOpen && item.verdict && (
                <div
                  className={`text-label font-bold px-1.5 py-0.5 rounded border shrink-0 ${styles.bg} ${styles.text} ${styles.border}`}
                >
                  {STATUS_SHORT[verdict]}
                </div>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
};