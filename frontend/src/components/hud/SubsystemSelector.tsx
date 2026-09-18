import React from 'react';
import { Layers, DoorOpen, Wind, Activity, GitCommit, ChevronLeft, ChevronRight } from 'lucide-react';
import { useTwinStore } from '../../store/useTwinStore';
import { SubsystemSelection } from '../../types/telemetry';

export const SubsystemSelector: React.FC = () => {
  const selectedSubsystem = useTwinStore((state) => state.selectedSubsystem);
  const setSelectedSubsystem = useTwinStore((state) => state.setSelectedSubsystem);
  const currentFrame = useTwinStore((state) => state.currentFrame);
  const isLeftDrawerOpen = useTwinStore((state) => state.isLeftDrawerOpen);
  const toggleLeftDrawer = useTwinStore((state) => state.toggleLeftDrawer);

  const subsystems = currentFrame?.subsystems;

  const items: {
    id: SubsystemSelection;
    label: string;
    sublabel: string;
    icon: React.FC<{ className?: string }>;
    score?: number;
  }[] = [
    {
      id: 'overview',
      label: 'SMRT Fleet Matrix',
      sublabel: 'Train-Track Infrastructure',
      icon: Layers,
    },
    {
      id: 'door',
      label: 'Door 3R System',
      sublabel: subsystems ? `${subsystems.door.cycle_state} • ${subsystems.door.motor_current_amps}A` : 'Electromechanics',
      icon: DoorOpen,
      score: subsystems?.door.anomaly_score,
    },
    {
      id: 'acv',
      label: 'ACV Climate Pack',
      sublabel: subsystems ? `ΔT ${subsystems.acv.delta_temp_c}°C • ${subsystems.acv.compressor_power_kw}kW` : 'Thermodynamics',
      icon: Wind,
      score: subsystems?.acv.anomaly_score,
    },
    {
      id: 'shm',
      label: 'Bogie SHM',
      sublabel: subsystems ? `${subsystems.shm.vibration_rms_g}g RMS • ${subsystems.shm.peak_frequency_hz}Hz` : 'Fatigue Dynamics',
      icon: Activity,
      score: subsystems?.shm.anomaly_score,
    },
    {
      id: 'rail',
      label: 'LTA Rail Infrastructure',
      sublabel: subsystems ? `${subsystems.rail_corrugation.depth_microns}μm • ${subsystems.rail_corrugation.wavelength_class}` : 'Track Corrugation',
      icon: GitCommit,
      score: subsystems?.rail_corrugation.severity_score,
    },
  ];

  return (
    <div className={`absolute top-20 left-4 z-20 transition-all duration-300 pointer-events-none ${isLeftDrawerOpen ? 'w-64' : 'w-12'}`}>
      {/* Drawer Container */}
      <div className="glass-panel p-2 rounded-2xl pointer-events-auto border-white/10 flex flex-col space-y-1.5 shadow-2xl">
        {/* Header with collapse button */}
        <div className={`flex items-center justify-between px-1 py-1 border-b border-white/5 mb-1 ${!isLeftDrawerOpen && 'justify-center'}`}>
          {isLeftDrawerOpen && (
            <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
              SMRT Subsystems
            </span>
          )}
          <button
            onClick={toggleLeftDrawer}
            className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800/80 transition-colors"
            title={isLeftDrawerOpen ? 'Collapse panel' : 'Expand panel'}
          >
            {isLeftDrawerOpen ? <ChevronLeft className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
          </button>
        </div>

        {/* Subsystem items */}
        {items.map((item) => {
          const isSelected = selectedSubsystem === item.id;
          const Icon = item.icon;
          const score = item.score ?? 0.0;
          const isAlert = score > 0.65;
          const isWarn = score > 0.40 && score <= 0.65;

          return (
            <button
              key={item.id}
              onClick={() => setSelectedSubsystem(item.id)}
              title={!isLeftDrawerOpen ? `${item.label} (${score ? (score * 100).toFixed(0) + '%' : 'Nominal'})` : undefined}
              className={`text-left transition-all rounded-xl border flex items-center ${
                isLeftDrawerOpen ? 'p-2.5 justify-between' : 'p-2 justify-center'
              } ${
                isSelected
                  ? 'border-red-600/70 bg-red-950/40 text-red-200 shadow-sm shadow-red-600/20'
                  : 'border-white/5 hover:border-white/15 hover:bg-slate-800/50 text-slate-300'
              }`}
            >
              <div className="flex items-center space-x-2.5">
                <div
                  className={`w-7 h-7 rounded-lg flex items-center justify-center transition-colors shrink-0 ${
                    isSelected
                      ? 'bg-red-600 text-white shadow-sm shadow-red-600/40'
                      : isAlert
                      ? 'bg-red-500/20 text-red-400'
                      : isWarn
                      ? 'bg-amber-500/20 text-amber-400'
                      : 'bg-slate-800/80 text-slate-400'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                </div>
                {isLeftDrawerOpen && (
                  <div className="min-w-0">
                    <div className="text-xs font-semibold text-slate-100 truncate">
                      {item.label}
                    </div>
                    <div className="text-[10px] text-slate-400 font-mono truncate max-w-[130px]">
                      {item.sublabel}
                    </div>
                  </div>
                )}
              </div>

              {/* Status pill (LTA green for OK, Amber for Warn, Red for Alert) */}
              {isLeftDrawerOpen && item.score !== undefined && (
                <div
                  className={`text-[9px] font-mono font-bold px-1.5 py-0.5 rounded border shrink-0 ${
                    isAlert
                      ? 'bg-red-500/20 text-red-300 border-red-500/40'
                      : isWarn
                      ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
                      : 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                  }`}
                >
                  {isAlert ? 'ALERT' : isWarn ? 'WARN' : 'NOMINAL'}
                </div>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
};
