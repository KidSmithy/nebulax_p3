/**
 * Plain-language glossary and verdict thresholds for live telemetry metrics.
 *
 * The good/warn/higherIsBetter values here must stay identical to
 * backend/core/config.py METRIC_THRESHOLDS - that Python table is the single
 * source of truth, this mirrors it so the UI's colour/verdict never
 * contradicts what the backend (and the AI insight service) already decided.
 */
import { MetricStatus } from '../types/telemetry';

export interface MetricDefinition {
  plainName: string;
  /** Shorter beginner label for tight layouts; falls back to plainName. */
  shortName?: string;
  technicalName: string;
  whatItIs: string;
  whyItMatters: string;
  analogy?: string;
  good: number;
  warn: number;
  higherIsBetter: boolean;
  unit: string;
  decimals: number;
}

export const METRIC_GLOSSARY: Record<string, MetricDefinition> = {
  'door.motor_current_amps': {
    plainName: 'Door motor effort',
    technicalName: 'Motor current',
    whatItIs: 'The electric current the door motor draws while it moves.',
    whyItMatters: 'A motor drawing more current than a healthy door needs is fighting extra friction somewhere in its mechanism.',
    analogy: 'Like pressing harder on a bike pedal because the chain is dragging.',
    good: 10.0,
    warn: 12.0,
    higherIsBetter: false,
    unit: 'A',
    decimals: 1,
  },
  'door.anomaly_score': {
    plainName: 'Door fault probability',
    shortName: 'Door risk',
    technicalName: 'Anomaly score',
    whatItIs: 'A model estimate of how likely this door is to be developing a fault, from 0 (healthy) to 1 (faulty).',
    whyItMatters: 'Catching a rising score early means a scheduled fix instead of a door that jams at a platform.',
    good: 0.4,
    warn: 0.65,
    higherIsBetter: false,
    unit: '',
    decimals: 2,
  },
  'door.transit_time_seconds': {
    plainName: 'Door travel time',
    technicalName: 'Transit time',
    whatItIs: 'How many seconds the door leaf takes to fully open or close.',
    whyItMatters: 'A slower cycle can hold the train at the platform and delay the whole line.',
    good: 3.3,
    warn: 3.8,
    higherIsBetter: false,
    unit: 's',
    decimals: 1,
  },
  'door.ghost_deviation_mm': {
    plainName: 'Door timing lag',
    technicalName: 'Ghost deviation',
    whatItIs: 'How far behind a healthy door\'s schedule this door leaf is, measured in millimetres of position.',
    whyItMatters: 'Growing lag is an early, physical sign of the same friction that later shows up as a fault.',
    good: 5.0,
    warn: 15.0,
    higherIsBetter: false,
    unit: 'mm',
    decimals: 1,
  },
  'acv.delta_temp_c': {
    plainName: 'Cooling achieved',
    technicalName: 'Delta-T',
    whatItIs: 'The temperature drop between the air going into the unit and the air it blows back into the cabin.',
    whyItMatters: 'A shrinking gap means passengers get a warmer cabin for the same electricity spent.',
    analogy: 'Like a fridge that used to chill a drink in a minute now taking three.',
    good: 7.5,
    warn: 6.0,
    higherIsBetter: true,
    unit: '°C',
    decimals: 1,
  },
  'acv.efficiency_rating': {
    plainName: 'Aircon efficiency',
    technicalName: 'Efficiency rating',
    whatItIs: 'How much cooling this unit produces per unit of electricity, where 1.0 is perfect efficiency.',
    whyItMatters: 'Falling efficiency means higher running costs for the same passenger comfort.',
    good: 0.8,
    warn: 0.65,
    higherIsBetter: true,
    unit: '',
    decimals: 2,
  },
  'acv.compressor_power_kw': {
    plainName: 'Compressor power draw',
    technicalName: 'Compressor power',
    whatItIs: 'The electricity the compressor is currently consuming.',
    whyItMatters: 'Rising power for the same cooling output usually means the unit is compensating for a fault.',
    good: 5.0,
    warn: 5.8,
    higherIsBetter: false,
    unit: 'kW',
    decimals: 1,
  },
  'acv.anomaly_score': {
    plainName: 'Aircon fault probability',
    shortName: 'Aircon risk',
    technicalName: 'Anomaly score',
    whatItIs: 'A model estimate of how likely this air-conditioning unit is to be developing a fault.',
    whyItMatters: 'Early warning here avoids a unit failing outright and losing cabin cooling entirely.',
    good: 0.35,
    warn: 0.55,
    higherIsBetter: false,
    unit: '',
    decimals: 2,
  },
  'shm.vibration_rms_g': {
    plainName: 'Wheel assembly shaking',
    technicalName: 'Axle-box vibration RMS',
    whatItIs: 'The average intensity of shaking measured at the axle box, in units of gravity (g).',
    whyItMatters: 'Sustained high vibration accelerates metal fatigue in the bogie frame and welds.',
    analogy: 'Like a washing machine on an unbalanced spin cycle - the shaking itself does wear and tear over time.',
    good: 2.5,
    warn: 3.5,
    higherIsBetter: false,
    unit: 'g',
    decimals: 2,
  },
  'shm.bearing_defect_prob': {
    plainName: 'Bearing damage probability',
    shortName: 'Bearing risk',
    technicalName: 'Bearing defect probability',
    whatItIs: 'How strongly the vibration spectrum shows the repeating knock pattern of a damaged wheel bearing.',
    whyItMatters: 'A bearing that seizes in service can force a train withdrawal, so this is watched closely.',
    good: 0.2,
    warn: 0.45,
    higherIsBetter: false,
    unit: '',
    decimals: 2,
  },
  'shm.fatigue_damage_index': {
    plainName: 'Accumulated metal fatigue',
    shortName: 'Metal fatigue',
    technicalName: 'Fatigue damage index',
    whatItIs: 'How much of the bogie\'s structural fatigue life has been used up, where 1.0 is end of life.',
    whyItMatters: 'This accumulates permanently - it never improves on its own, only maintenance resets the part it applies to.',
    good: 0.4,
    warn: 0.7,
    higherIsBetter: false,
    unit: '',
    decimals: 2,
  },
  'shm.anomaly_score': {
    plainName: 'Bogie fault probability',
    shortName: 'Bogie risk',
    technicalName: 'Anomaly score',
    whatItIs: 'A model estimate of how likely the bogie assembly is to be developing a fault.',
    whyItMatters: 'Combines several structural signals into one early-warning number.',
    good: 0.5,
    warn: 0.7,
    higherIsBetter: false,
    unit: '',
    decimals: 2,
  },
  'rail_corrugation.depth_microns': {
    plainName: 'Rail ripple depth',
    technicalName: 'Corrugation depth',
    whatItIs: 'How deep the ripples worn into the top of the rail are, in micrometres.',
    whyItMatters: 'Deeper ripples hammer the wheels harder every time the train passes over them.',
    analogy: 'Like the depth of ridges on a washboard road surface.',
    good: 20.0,
    warn: 35.0,
    higherIsBetter: false,
    unit: 'μm',
    decimals: 0,
  },
  'rail_corrugation.severity_score': {
    plainName: 'Track condition severity',
    shortName: 'Track severity',
    technicalName: 'Severity score',
    whatItIs: 'A combined 0-1 score for how badly this stretch of rail needs maintenance.',
    whyItMatters: 'Used to prioritise which sections get ground first across the whole line.',
    good: 0.35,
    warn: 0.65,
    higherIsBetter: false,
    unit: '',
    decimals: 2,
  },
  fleet_health_index: {
    plainName: 'Overall train health',
    technicalName: 'Fleet health index',
    whatItIs: 'A single 0-100% score combining the door, air-conditioning, bogie and track readings for this train.',
    whyItMatters: 'The one number to check first before drilling into any individual subsystem.',
    good: 0.8,
    warn: 0.6,
    higherIsBetter: true,
    unit: '',
    decimals: 2,
  },
  track_chainage_km: {
    plainName: 'Track position',
    technicalName: 'Chainage',
    whatItIs: 'The train\'s position along the line, measured in kilometre posts (KP) from the line\'s origin.',
    whyItMatters: 'Corrugation zones and other track defects are tied to fixed KP locations, so position tells you what is coming up.',
    good: 9999,
    warn: 9999,
    higherIsBetter: true,
    unit: 'km',
    decimals: 3,
  },
};

export const STATUS_STYLES: Record<MetricStatus, { bg: string; text: string; border: string; bar: string }> = {
  GOOD: { bg: 'bg-emerald-50', text: 'text-emerald-700', border: 'border-emerald-200', bar: 'bg-emerald-500' },
  WATCH: { bg: 'bg-amber-50', text: 'text-amber-700', border: 'border-amber-200', bar: 'bg-amber-500' },
  ACTION_NEEDED: { bg: 'bg-red-50', text: 'text-red-700', border: 'border-red-200', bar: 'bg-red-500' },
  UNKNOWN: { bg: 'bg-slate-50', text: 'text-slate-500', border: 'border-slate-200', bar: 'bg-slate-400' },
};

export const STATUS_SHORT: Record<MetricStatus, string> = {
  GOOD: 'OK',
  WATCH: 'WATCH',
  ACTION_NEEDED: 'ACT NOW',
  UNKNOWN: 'N/A',
};

/** Mirrors backend/core/config.py classify_metric(). */
export function classifyMetric(path: string, value: number | null | undefined): MetricStatus {
  const spec = METRIC_GLOSSARY[path];
  if (!spec || value === null || value === undefined || Number.isNaN(value)) return 'UNKNOWN';
  const { good, warn, higherIsBetter } = spec;
  if (higherIsBetter) {
    if (value >= good) return 'GOOD';
    return value >= warn ? 'WATCH' : 'ACTION_NEEDED';
  }
  if (value <= good) return 'GOOD';
  return value <= warn ? 'WATCH' : 'ACTION_NEEDED';
}

export function formatMetric(path: string, value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  const def = METRIC_GLOSSARY[path];
  if (!def) return String(value);
  const num = value.toFixed(def.decimals);
  return def.unit ? `${num} ${def.unit}` : num;
}

/**
 * Normalises a value to [0, 1] for the gauge bar, increasing monotonically
 * with the raw value regardless of whether the metric is higher-is-better.
 * Both threshold markers land inside the visible range with margin on either
 * side for out-of-band values.
 */
export function gaugePosition(path: string, value: number): number {
  const def = METRIC_GLOSSARY[path];
  if (!def) return 0;
  const { good, warn } = def;
  const span = Math.abs(warn - good) || 1;
  const lo = Math.min(good, warn) - span;
  const hi = Math.max(good, warn) + span;
  const pos = (value - lo) / (hi - lo);
  return Math.max(0, Math.min(1, pos));
}

interface TermInfo {
  plain: string;
  explanation: string;
}

const TERM_INFO: Record<string, TermInfo> = {
  // Door cycle_state
  IDLE: { plain: 'Idle', explanation: 'The door is closed and not currently cycling.' },
  OPENING: { plain: 'Opening', explanation: 'The door leaf is currently travelling open.' },
  DWELL_OPEN: { plain: 'Holding open', explanation: 'The door is open for passenger boarding and alighting.' },
  CLOSING: { plain: 'Closing', explanation: 'The door leaf is currently travelling closed.' },
  CLOSED_LOCKED: { plain: 'Closed & locked', explanation: 'The door is closed and mechanically locked for travel.' },

  // Door / ACV fault_type
  NONE: { plain: 'No fault detected', explanation: 'This subsystem is operating within its normal range.' },
  GUIDE_RAIL_FRICTION: {
    plain: 'Guide rail friction',
    explanation: "The door leaf is dragging against its guide rail, making the motor work harder than a healthy door would.",
  },
  ROLLER_BEARING_WEAR: {
    plain: 'Roller bearing wear',
    explanation: "A worn roller bearing is adding resistance to the door's motion, showing up as extra current draw and jitter.",
  },
  REFRIGERANT_LEAKAGE: {
    plain: 'Refrigerant leakage',
    explanation: 'A drop in refrigerant charge is reducing how much heat the unit can remove per unit of electricity.',
  },
  FILTER_CLOGGING: {
    plain: 'Filter clogging',
    explanation: 'A blocked air filter is restricting airflow, forcing the compressor to work harder for less cooling.',
  },

  // Rail wavelength_class
  SMOOTH_GROUND: {
    plain: 'Recently ground smooth',
    explanation: 'This section of rail has recently been ground and shows minimal surface roughness.',
  },
  NOMINAL: {
    plain: 'Normal surface',
    explanation: 'The rail surface in this section is within its normal roughness range.',
  },
  SHORT_PITCH: {
    plain: 'Short-pitch corrugation',
    explanation: 'Closely spaced ripples worn into the rail head that produce a high-pitched rumble.',
  },
  LONG_PITCH: {
    plain: 'Long-pitch corrugation',
    explanation: 'Widely spaced ripples that produce a lower-frequency shaking.',
  },

  // SHM critical_weld_node
  BOLSTER_RIB_L2: {
    plain: 'Bolster rib, left, node 2',
    explanation: 'The second stiffening rib on the left side of the bogie bolster, the cross-member that carries the carriage weight onto the wheelset.',
  },
  SIDE_BEAM_R1: {
    plain: 'Side beam, right, node 1',
    explanation: 'The first weld joint along the right-hand side beam of the bogie frame.',
  },
};

function prettifyRaw(raw: string): string {
  return raw
    .toLowerCase()
    .split('_')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}

/** Short plain-language label for a raw enum-like telemetry string. */
export function plainTerm(raw: string | undefined | null): string {
  if (!raw) return '—';
  return TERM_INFO[raw]?.plain ?? prettifyRaw(raw);
}

/** Plain name + one-line explanation for a raw enum-like telemetry string. */
export function explainTerm(
  raw: string | undefined | null
): { plainName: string; explanation: string } | undefined {
  if (!raw) return undefined;
  const info = TERM_INFO[raw];
  if (info) return { plainName: info.plain, explanation: info.explanation };
  return { plainName: prettifyRaw(raw), explanation: 'No further detail available for this reading.' };
}
