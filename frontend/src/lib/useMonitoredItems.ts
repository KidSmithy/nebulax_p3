import {
  Activity,
  DoorOpen,
  GitCommit,
  Layers,
  Wind,
} from 'lucide-react';
import React from 'react';
import { useTwinStore } from '../store/useTwinStore';
import { InterventionAction, MetricStatus, SubsystemSelection } from '../types/telemetry';
import { classifyMetric, plainTerm } from './metricGlossary';

const VERDICT_RANK: Record<MetricStatus, number> = {
  ACTION_NEEDED: 3,
  WATCH: 2,
  GOOD: 1,
  UNKNOWN: 0,
};

/** Which repair action fixes which subsystem's single canonical row. */
export const REPAIR_FOR_SUBSYSTEM: Partial<Record<SubsystemSelection, { action: InterventionAction; label: string }>> = {
  acv: { action: 'ACTION_REPLACE_FILTER', label: 'Replace the aircon filter' },
  door: { action: 'ACTION_LUBRICATE_DOOR', label: 'Grease the door tracks' },
  shm: { action: 'ACTION_INSPECT_BEARING', label: 'Service the wheel bearing' },
};

export interface MonitoredItem {
  id: SubsystemSelection;
  plainLabel: string;
  technicalLabel: string;
  plainSub: string;
  technicalSub: string;
  icon: React.FC<{ className?: string }>;
  verdict?: MetricStatus;
}

/**
 * The single alarm system's ordering: worst subsystem first (only promoted -
 * spine + repair link - if it's genuinely not GOOD), "Whole train" always
 * last since it's a nav item, not a status row. Shared by SubsystemSelector
 * and the Conductor's compact views so they never drift out of sync.
 */
export function useMonitoredItems() {
  const currentFrame = useTwinStore((state) => state.currentFrame);
  const subsystems = currentFrame?.subsystems;
  const status = currentFrame?.plain_status ?? {};

  const items: MonitoredItem[] = [
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
      plainLabel: 'Air conditioning',
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
      plainLabel: 'Structural health monitoring (SHM)',
      technicalLabel: 'Bogie SHM',
      plainSub: subsystems
        ? `shaking ${subsystems.shm.vibration_rms_g.toFixed(2)} g`
        : 'Structural health',
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
      plainLabel: 'Rail corrugation',
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

  const overviewItem = items.find((i) => i.id === 'overview')!;
  const subsystemItems = items
    .filter((i) => i.id !== 'overview')
    .sort((a, b) => VERDICT_RANK[b.verdict ?? 'UNKNOWN'] - VERDICT_RANK[a.verdict ?? 'UNKNOWN']);
  const orderedItems = [...subsystemItems, overviewItem];
  const topVerdict = subsystemItems[0]?.verdict ?? 'UNKNOWN';
  const isPromoted = (id: SubsystemSelection, idx: number) =>
    idx === 0 && id !== 'overview' && VERDICT_RANK[topVerdict] > VERDICT_RANK.GOOD;

  return { orderedItems, subsystemItems, topVerdict, isPromoted, VERDICT_RANK };
}
