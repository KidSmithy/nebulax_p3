import React, { useState } from 'react';
import { useFrame } from '@react-three/fiber';
import { Html, Line } from '@react-three/drei';
import { Activity, DoorOpen, Wind, GitCommit, X, ExternalLink } from 'lucide-react';
import { useTwinStore } from '../../store/useTwinStore';
import { carShiftZ } from './consist';
import { MetricStatus } from '../../types/telemetry';
import { STATUS_SHORT, STATUS_STYLES } from '../../lib/metricGlossary';

interface InspectionTagProps {
  type: 'door' | 'acv' | 'shm' | 'rail';
  position: [number, number, number];
  beaconLabel: string;
  /** Drives the leader-line dot colour and the idle pill's status dot. */
  status?: MetricStatus;
}

/** World-space offset from the anchor to where the pill itself floats. */
const PILL_OFFSET: [number, number, number] = [0.32, 0.42, 0.12];

const STATUS_DOT_COLOR: Record<MetricStatus, string> = {
  GOOD: '#0f766e',
  WATCH: '#B45309',
  ACTION_NEEDED: '#ED1C24',
  UNKNOWN: '#94a3b8',
};

export const InspectionTag: React.FC<InspectionTagProps> = ({
  type,
  position,
  beaconLabel,
  status = 'UNKNOWN',
}) => {
  const activeInspection = useTwinStore((state) => state.activeInspection);
  const setActiveInspection = useTwinStore((state) => state.setActiveInspection);
  const setSelectedSubsystem = useTwinStore((state) => state.setSelectedSubsystem);
  const toggleRightDrawer = useTwinStore((state) => state.toggleRightDrawer);
  const setRightDrawerTab = useTwinStore((state) => state.setRightDrawerTab);
  const currentFrame = useTwinStore((state) => state.currentFrame);
  const isHudVisible = useTwinStore((state) => state.isHudVisible);
  // Tags are sized for close inspection; zoomed far out they stack into an
  // unreadable pile, so they step aside (measured from the monitored car).
  const monitoredCar = useTwinStore((state) => state.monitoredCar);
  const [farAway, setFarAway] = useState(false);
  useFrame(({ camera }) => {
    const { x, y, z } = camera.position;
    const far = Math.hypot(x, y, z - carShiftZ(monitoredCar)) > 70;
    if (far !== farAway) setFarAway(far);
  });

  if (!isHudVisible || farAway) return null;

  const isActive = activeInspection === type;
  const subsystems = currentFrame?.subsystems;
  const dotColor = STATUS_DOT_COLOR[status];

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

  const statusStyles = STATUS_STYLES[status];

  return (
    <group position={position}>
      {/* Ringed anchor dot on the part itself, plus a 1px leader line up to the pill. */}
      <mesh>
        <ringGeometry args={[0.028, 0.045, 24]} />
        <meshBasicMaterial color={dotColor} toneMapped={false} />
      </mesh>
      <mesh>
        <circleGeometry args={[0.018, 16]} />
        <meshBasicMaterial color={dotColor} toneMapped={false} />
      </mesh>
      <Line points={[[0, 0, 0], PILL_OFFSET]} color="#dee0d8" lineWidth={1} />

      <group position={PILL_OFFSET}>
        <Html center style={{ zIndex: isActive ? 1000 : 10 }}>
          {!isActive ? (
            /* Light annotation pill: white, hairline outline, mono label, status dot. */
            <button
              onClick={handleClickBeacon}
              className="flex items-center space-x-1.5 px-2 py-1 rounded-full bg-white border border-slate-200 text-ink-900 hover:border-ink-500 transition-colors duration-150 cursor-pointer select-none glass-panel-floating"
            >
              <span
                className="w-1.5 h-1.5 rounded-full shrink-0"
                style={{ backgroundColor: dotColor }}
                aria-hidden="true"
              />
              <span className="text-label font-mono font-semibold whitespace-nowrap">
                {beaconLabel}
              </span>
            </button>
          ) : (
            /* Active Expanded In-Scene Spatial Stats Card */
            <div
              onClick={(e) => e.stopPropagation()}
              className="w-64 glass-panel-floating p-3 rounded text-slate-800 text-xs font-sans pointer-events-auto select-none animate-in fade-in zoom-in-95 duration-150 relative z-50"
            >
              {/* Header */}
              <div className="flex items-center justify-between border-b border-slate-100 pb-1.5 mb-2">
                <div className="flex items-center space-x-1.5">
                  {type === 'shm' && <Activity className="w-4 h-4 text-ink-700" />}
                  {type === 'door' && <DoorOpen className="w-4 h-4 text-ink-700" />}
                  {type === 'acv' && <Wind className="w-4 h-4 text-ink-700" />}
                  {type === 'rail' && <GitCommit className="w-4 h-4 text-ink-700" />}
                  <span className="font-bold text-label text-slate-900">
                    {beaconLabel}
                  </span>
                </div>
                <div className="flex items-center space-x-1.5">
                  <span
                    className={`px-1.5 py-0.5 rounded-full text-label font-semibold border ${statusStyles.bg} ${statusStyles.text} ${statusStyles.border}`}
                  >
                    {STATUS_SHORT[status]}
                  </span>
                  <button
                    onClick={handleClose}
                    className="p-1 rounded text-slate-400 hover:text-slate-800 hover:bg-slate-100 transition-colors"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>

              {/* Subsystem Real-time Stats Grid */}
              {type === 'shm' && subsystems?.shm && (
                <div className="space-y-1.5 font-mono text-label">
                  <div className="grid grid-cols-2 gap-1.5">
                    <div className="bg-white p-1.5 rounded border border-slate-200">
                      <div className="text-slate-500 text-[8px] font-semibold">Vibration RMS</div>
                      <div className="text-ink-900 font-bold text-xs">{subsystems.shm.vibration_rms_g} g</div>
                    </div>
                    <div className="bg-white p-1.5 rounded border border-slate-200">
                      <div className="text-slate-500 text-[8px] font-semibold">Peak Freq</div>
                      <div className="text-slate-900 font-bold text-xs">{subsystems.shm.peak_frequency_hz} Hz</div>
                    </div>
                  </div>
                  <div className="bg-white p-1.5 rounded border border-slate-200 flex justify-between items-center">
                    <span className="text-slate-500">Critical Node:</span>
                    <span className="text-ink-900 font-bold">{subsystems.shm.critical_weld_node}</span>
                  </div>
                  <div className="bg-white p-1.5 rounded border border-slate-200 flex justify-between items-center">
                    <span className="text-slate-500">Fatigue Damage:</span>
                    <span className="text-slate-900 font-bold">{(subsystems.shm.fatigue_damage_index * 100).toFixed(0)}%</span>
                  </div>
                </div>
              )}

              {type === 'door' && subsystems?.door && (
                <div className="space-y-1.5 font-mono text-label">
                  <div className="grid grid-cols-2 gap-1.5">
                    <div className="bg-white p-1.5 rounded border border-slate-200">
                      <div className="text-slate-500 text-[8px] font-semibold">Motor Current</div>
                      <div className="text-ink-900 font-bold text-xs">{subsystems.door.motor_current_amps} A</div>
                    </div>
                    <div className="bg-white p-1.5 rounded border border-slate-200">
                      <div className="text-slate-500 text-[8px] font-semibold">Cycle State</div>
                      <div className="text-slate-900 font-bold text-xs">{subsystems.door.cycle_state}</div>
                    </div>
                  </div>
                  <div className="bg-white p-1.5 rounded border border-slate-200 flex justify-between items-center">
                    <span className="text-slate-500">Transit Duration:</span>
                    <span className="text-slate-900 font-bold">{subsystems.door.transit_time_seconds}s</span>
                  </div>
                  <div className="bg-white p-1.5 rounded border border-slate-200 flex justify-between items-center">
                    <span className="text-slate-500">Ghost Lag Δx:</span>
                    <span className="text-status-nominal font-bold">{subsystems.door.ghost_deviation_mm} mm</span>
                  </div>
                </div>
              )}

              {type === 'acv' && subsystems?.acv && (
                <div className="space-y-1.5 font-mono text-label">
                  <div className="grid grid-cols-2 gap-1.5">
                    <div className="bg-white p-1.5 rounded border border-slate-200">
                      <div className="text-slate-500 text-[8px] font-semibold">Delta-T</div>
                      <div className="text-ink-900 font-bold text-xs">{subsystems.acv.delta_temp_c} °C</div>
                    </div>
                    <div className="bg-white p-1.5 rounded border border-slate-200">
                      <div className="text-slate-500 text-[8px] font-semibold">Power</div>
                      <div className="text-slate-900 font-bold text-xs">{subsystems.acv.compressor_power_kw} kW</div>
                    </div>
                  </div>
                  <div className="bg-white p-1.5 rounded border border-slate-200 flex justify-between items-center">
                    <span className="text-slate-500">COP Rating:</span>
                    <span className="text-status-nominal font-bold">{subsystems.acv.efficiency_rating}</span>
                  </div>
                  <div className="bg-white p-1.5 rounded border border-slate-200 flex justify-between items-center">
                    <span className="text-slate-500">Refrigerant Leak:</span>
                    <span className={subsystems.acv.fault_type === 'NONE' ? 'text-status-nominal font-semibold' : 'text-status-fault font-semibold'}>
                      {subsystems.acv.fault_type}
                    </span>
                  </div>
                </div>
              )}

              {type === 'rail' && subsystems?.rail_corrugation && (
                <div className="space-y-1.5 font-mono text-label">
                  <div className="grid grid-cols-2 gap-1.5">
                    <div className="bg-white p-1.5 rounded border border-slate-200">
                      <div className="text-slate-500 text-[8px] font-semibold">Roughness</div>
                      <div className="text-status-watch font-bold text-xs">{subsystems.rail_corrugation.depth_microns} μm</div>
                    </div>
                    <div className="bg-white p-1.5 rounded border border-slate-200">
                      <div className="text-slate-500 text-[8px] font-semibold">Wavelength</div>
                      <div className="text-slate-900 font-bold text-xs truncate">{subsystems.rail_corrugation.wavelength_class}</div>
                    </div>
                  </div>
                  <div className="bg-white p-1.5 rounded border border-slate-200 flex justify-between items-center">
                    <span className="text-slate-500">Grinding Urgency:</span>
                    <span className="text-status-nominal font-bold">{subsystems.rail_corrugation.maintenance_urgency}</span>
                  </div>
                </div>
              )}

              {/* Footer action link - quiet, not a solid button */}
              <div className="mt-2.5 pt-2 border-t border-slate-100">
                <button
                  onClick={handleOpenCockpit}
                  className="w-full text-left font-mono text-label font-semibold text-ink-700 hover:text-ink-900 flex items-center justify-between transition-colors duration-150"
                >
                  <span>Open cockpit charts</span>
                  <ExternalLink className="w-3 h-3" />
                </button>
              </div>
            </div>
          )}
        </Html>
      </group>
    </group>
  );
};
