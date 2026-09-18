import React from 'react';
import { Html } from '@react-three/drei';
import { Activity, DoorOpen, Wind, GitCommit, X, ExternalLink } from 'lucide-react';
import { useTwinStore } from '../../store/useTwinStore';

interface InspectionTagProps {
  type: 'door' | 'acv' | 'shm' | 'rail';
  position: [number, number, number];
  beaconLabel: string;
}

export const InspectionTag: React.FC<InspectionTagProps> = ({ type, position, beaconLabel }) => {
  const activeInspection = useTwinStore((state) => state.activeInspection);
  const setActiveInspection = useTwinStore((state) => state.setActiveInspection);
  const setSelectedSubsystem = useTwinStore((state) => state.setSelectedSubsystem);
  const toggleRightDrawer = useTwinStore((state) => state.toggleRightDrawer);
  const setRightDrawerTab = useTwinStore((state) => state.setRightDrawerTab);
  const currentFrame = useTwinStore((state) => state.currentFrame);
  const isHudVisible = useTwinStore((state) => state.isHudVisible);

  if (!isHudVisible) return null;

  const isActive = activeInspection === type;
  const subsystems = currentFrame?.subsystems;

  const handleOpenCockpit = (e: React.MouseEvent) => {
    e.stopPropagation();
    setSelectedSubsystem(type);
    setRightDrawerTab('telemetry');
    useTwinStore.setState({ isRightDrawerOpen: true });
  };

  const handleClose = (e: React.MouseEvent) => {
    e.stopPropagation();
    setActiveInspection(null);
  };

  const handleClickBeacon = (e: React.MouseEvent) => {
    e.stopPropagation();
    setActiveInspection(isActive ? null : type);
  };

  const isAnyActive = Boolean(activeInspection);

  // If another inspection is open, hide this beacon pin so it doesn't clutter or overlap
  if (isAnyActive && !isActive) return null;

  return (
    <group position={position}>
      <Html center style={{ zIndex: isActive ? 1000 : 10 }}>
        {!isActive ? (
          /* Sleek Compact 3D Clickable Beacon Pin */
          <button
            onClick={handleClickBeacon}
            className="group flex items-center space-x-1.5 px-2 py-0.5 rounded-full bg-slate-950/85 backdrop-blur-md border border-red-500/50 text-slate-200 hover:border-red-400 hover:bg-red-950/90 shadow-[0_0_12px_rgba(237,28,36,0.35)] transition-all hover:scale-105 cursor-pointer select-none"
          >
            <span className="relative flex h-2 w-2 shrink-0">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-red-500"></span>
            </span>
            <span className="text-[10px] font-mono font-bold tracking-wider uppercase text-slate-100 whitespace-nowrap">
              {beaconLabel}
            </span>
          </button>
        ) : (
          /* Active Expanded In-Scene Spatial Stats Card */
          <div
            onClick={(e) => e.stopPropagation()}
            className="w-64 glass-panel p-3 rounded-2xl border-red-500/50 shadow-2xl text-slate-100 text-xs font-sans pointer-events-auto select-none animate-in fade-in zoom-in-95 duration-150 relative z-50"
          >
            {/* Header */}
            <div className="flex items-center justify-between border-b border-white/10 pb-1.5 mb-2">
              <div className="flex items-center space-x-1.5">
                {type === 'shm' && <Activity className="w-4 h-4 text-red-500" />}
                {type === 'door' && <DoorOpen className="w-4 h-4 text-red-500" />}
                {type === 'acv' && <Wind className="w-4 h-4 text-sky-400" />}
                {type === 'rail' && <GitCommit className="w-4 h-4 text-amber-400" />}
                <span className="font-bold uppercase tracking-wider text-[11px] text-white">
                  {beaconLabel}
                </span>
              </div>
              <div className="flex items-center space-x-1.5">
                <span
                  className={`px-1.5 py-0.2 rounded text-[8px] font-mono font-bold border ${
                    type === 'shm' && (subsystems?.shm?.fatigue_damage_index ?? 0) > 0.35
                      ? 'bg-amber-500/15 text-amber-400 border-amber-500/40'
                      : type === 'door' && (subsystems?.door?.anomaly_score ?? 0) > 0.4
                      ? 'bg-red-500/15 text-red-400 border-red-500/40'
                      : 'bg-emerald-500/15 text-emerald-400 border-emerald-500/40'
                  }`}
                >
                  {type === 'shm' && (subsystems?.shm?.fatigue_damage_index ?? 0) > 0.35
                    ? 'WARN'
                    : type === 'door' && (subsystems?.door?.anomaly_score ?? 0) > 0.4
                    ? 'FAULT'
                    : 'NOMINAL'}
                </span>
                <button
                  onClick={handleClose}
                  className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>

            {/* Subsystem Real-time Stats Grid */}
            {type === 'shm' && subsystems?.shm && (
              <div className="space-y-1.5 font-mono text-[10px]">
                <div className="grid grid-cols-2 gap-1.5">
                  <div className="bg-dark-850/80 p-1.5 rounded-lg border border-white/5">
                    <div className="text-slate-400 text-[8px] uppercase">Vibration RMS</div>
                    <div className="text-red-400 font-bold text-xs">{subsystems.shm.vibration_rms_g} g</div>
                  </div>
                  <div className="bg-dark-850/80 p-1.5 rounded-lg border border-white/5">
                    <div className="text-slate-400 text-[8px] uppercase">Peak Freq</div>
                    <div className="text-white font-bold text-xs">{subsystems.shm.peak_frequency_hz} Hz</div>
                  </div>
                </div>
                <div className="bg-dark-850/80 p-1.5 rounded-lg border border-white/5 flex justify-between items-center">
                  <span className="text-slate-400">Critical Node:</span>
                  <span className="text-amber-400 font-bold">{subsystems.shm.critical_weld_node}</span>
                </div>
                <div className="bg-dark-850/80 p-1.5 rounded-lg border border-white/5 flex justify-between items-center">
                  <span className="text-slate-400">Fatigue Damage:</span>
                  <span className="text-white font-bold">{(subsystems.shm.fatigue_damage_index * 100).toFixed(0)}%</span>
                </div>
              </div>
            )}

            {type === 'door' && subsystems?.door && (
              <div className="space-y-1.5 font-mono text-[10px]">
                <div className="grid grid-cols-2 gap-1.5">
                  <div className="bg-dark-850/80 p-1.5 rounded-lg border border-white/5">
                    <div className="text-slate-400 text-[8px] uppercase">Motor Current</div>
                    <div className="text-red-400 font-bold text-xs">{subsystems.door.motor_current_amps} A</div>
                  </div>
                  <div className="bg-dark-850/80 p-1.5 rounded-lg border border-white/5">
                    <div className="text-slate-400 text-[8px] uppercase">Cycle State</div>
                    <div className="text-white font-bold text-xs">{subsystems.door.cycle_state}</div>
                  </div>
                </div>
                <div className="bg-dark-850/80 p-1.5 rounded-lg border border-white/5 flex justify-between items-center">
                  <span className="text-slate-400">Transit Duration:</span>
                  <span className="text-white font-bold">{subsystems.door.transit_time_seconds}s</span>
                </div>
                <div className="bg-dark-850/80 p-1.5 rounded-lg border border-white/5 flex justify-between items-center">
                  <span className="text-slate-400">Ghost Lag Δx:</span>
                  <span className="text-emerald-400 font-bold">{subsystems.door.ghost_deviation_mm} mm</span>
                </div>
              </div>
            )}

            {type === 'acv' && subsystems?.acv && (
              <div className="space-y-1.5 font-mono text-[10px]">
                <div className="grid grid-cols-2 gap-1.5">
                  <div className="bg-dark-850/80 p-1.5 rounded-lg border border-white/5">
                    <div className="text-slate-400 text-[8px] uppercase">Delta-T</div>
                    <div className="text-sky-400 font-bold text-xs">{subsystems.acv.delta_temp_c} °C</div>
                  </div>
                  <div className="bg-dark-850/80 p-1.5 rounded-lg border border-white/5">
                    <div className="text-slate-400 text-[8px] uppercase">Power</div>
                    <div className="text-white font-bold text-xs">{subsystems.acv.compressor_power_kw} kW</div>
                  </div>
                </div>
                <div className="bg-dark-850/80 p-1.5 rounded-lg border border-white/5 flex justify-between items-center">
                  <span className="text-slate-400">COP Rating:</span>
                  <span className="text-emerald-400 font-bold">{subsystems.acv.efficiency_rating}</span>
                </div>
                <div className="bg-dark-850/80 p-1.5 rounded-lg border border-white/5 flex justify-between items-center">
                  <span className="text-slate-400">Refrigerant Leak:</span>
                  <span className={subsystems.acv.fault_type === 'NONE' ? 'text-emerald-400' : 'text-red-400'}>
                    {subsystems.acv.fault_type}
                  </span>
                </div>
              </div>
            )}

            {type === 'rail' && subsystems?.rail_corrugation && (
              <div className="space-y-1.5 font-mono text-[10px]">
                <div className="grid grid-cols-2 gap-1.5">
                  <div className="bg-dark-850/80 p-1.5 rounded-lg border border-white/5">
                    <div className="text-slate-400 text-[8px] uppercase">Roughness</div>
                    <div className="text-amber-400 font-bold text-xs">{subsystems.rail_corrugation.depth_microns} μm</div>
                  </div>
                  <div className="bg-dark-850/80 p-1.5 rounded-lg border border-white/5">
                    <div className="text-slate-400 text-[8px] uppercase">Wavelength</div>
                    <div className="text-white font-bold text-xs truncate">{subsystems.rail_corrugation.wavelength_class}</div>
                  </div>
                </div>
                <div className="bg-dark-850/80 p-1.5 rounded-lg border border-white/5 flex justify-between items-center">
                  <span className="text-slate-400">Grinding Urgency:</span>
                  <span className="text-emerald-400 font-bold">{subsystems.rail_corrugation.maintenance_urgency}</span>
                </div>
              </div>
            )}

            {/* Footer action button */}
            <div className="mt-2.5 pt-2 border-t border-white/10 flex items-center justify-between">
              <button
                onClick={handleOpenCockpit}
                className="w-full py-1 px-2.5 rounded-lg bg-red-600 hover:bg-red-500 text-white font-mono text-[10px] font-bold flex items-center justify-center space-x-1 transition-all shadow-md shadow-red-600/30"
              >
                <span>OPEN COCKPIT CHARTS</span>
                <ExternalLink className="w-3 h-3 ml-1" />
              </button>
            </div>
          </div>
        )}
      </Html>
    </group>
  );
};
