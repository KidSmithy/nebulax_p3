export interface FFTPoint {
  freq_hz: number;
  amp: number;
}

export interface DoorTelemetry {
  active_door_id: string;
  cycle_state: string; // 'OPENING' | 'CLOSING' | 'DWELL_OPEN' | 'CLOSED_LOCKED'
  transit_time_seconds: number;
  motor_current_amps: number;
  nominal_current_amps: number;
  anomaly_score: number;
  fault_type: string;
  ghost_deviation_mm: number;
  waveform_window: number[];
}

export interface ACVTelemetry {
  unit_id: string;
  supply_temp_c: number;
  return_temp_c: number;
  delta_temp_c: number;
  compressor_power_kw: number;
  efficiency_rating: number;
  anomaly_score: number;
  fault_type: string;
}

export interface SHMTelemetry {
  bogie_id: string;
  vibration_rms_g: number;
  peak_frequency_hz: number;
  bearing_defect_prob: number;
  bearing_defect_freq_hz?: number;
  fatigue_damage_index: number;
  anomaly_score: number;
  critical_weld_node: string;
  fft_spectrum: FFTPoint[];
}

export interface RailCorrugationTelemetry {
  kp_start: number;
  kp_end: number;
  depth_microns: number;
  wavelength_class: string;
  severity_score: number;
  maintenance_urgency: string;
  grinding_priority_rank: number;
}

export interface SubsystemsPayload {
  door: DoorTelemetry;
  acv: ACVTelemetry;
  shm: SHMTelemetry;
  rail_corrugation: RailCorrugationTelemetry;
}

/** Plain-language verdict shared by backend and UI. */
export type MetricStatus = 'GOOD' | 'WATCH' | 'ACTION_NEEDED' | 'UNKNOWN';

export type InterventionAction =
  | 'ACTION_GRIND_RAIL'
  | 'ACTION_LUBRICATE_DOOR'
  | 'ACTION_REPLACE_FILTER'
  | 'ACTION_INSPECT_BEARING';

export type ActiveInterventions = Record<InterventionAction, boolean>;

/** One metric measured before vs after the active maintenance actions. */
export interface CounterfactualMetric {
  path: string;
  label: string;
  unit: string;
  decimals: number;
  before: number;
  after: number;
  delta: number;
  percent_change: number | null;
  direction: 'better' | 'worse' | 'unchanged';
  status_before: MetricStatus;
  status_after: MetricStatus;
}

export interface CounterfactualBlock {
  active: boolean;
  active_actions: string[];
  metrics: CounterfactualMetric[];
  improved_count?: number;
  headline: string | null;
}

export interface CorrugationZoneInfo {
  kp_start: number;
  kp_end: number;
  kp_centre: number;
  severity: number;
  wavelength_class: string;
  urgency: string;
  distance_km: number;
  is_inside_zone: boolean;
}

export interface UnifiedTelemetryFrame {
  type?: string;
  frame_id: number;
  timestamp: string;
  track_chainage_km: number;
  train_speed_kmh: number;
  fleet_health_index: number;
  subsystems: SubsystemsPayload;
  active_interventions?: ActiveInterventions;
  plain_status?: Record<string, MetricStatus>;
  counterfactual?: CounterfactualBlock;
  next_corrugation_zone?: CorrugationZoneInfo;
}

/** Impact measurement echoed back when an intervention is toggled. */
export interface WhatIfResult {
  type?: string;
  action?: string;
  enabled?: boolean;
  title?: string;
  plain_description?: string;
  technical_description?: string;
  status?: string;
  wired_to_simulation?: boolean;
  active_interventions?: ActiveInterventions;
  measured_impact?: {
    has_measurable_effect: boolean;
    changed_metrics: Array<{
      path: string;
      label: string;
      unit: string;
      decimals: number;
      before: number;
      after: number;
      delta: number;
      direction: 'better' | 'worse';
    }>;
    note: string | null;
    measured_at_peak_travel?: boolean;
  };
  next_corrugation_zone?: CorrugationZoneInfo;
}

export interface AIFinding {
  subsystem: string;
  what_it_means: string;
  why_it_matters: string;
}

export interface AIInsight {
  headline: string;
  severity: MetricStatus;
  summary: string;
  findings: AIFinding[];
  recommended_action: string;
  urgency: 'NONE' | 'ROUTINE' | 'WITHIN_7_DAYS' | 'WITHIN_48_HOURS';
  analogy?: string;
  source: 'openai' | 'rule_based' | string;
  model?: string;
  degraded_reason?: string | null;
  stale?: boolean;
}

export interface AIAnswer {
  question: string;
  answer: string;
  source: string;
  model?: string;
  degraded_reason?: string | null;
}

export type CameraPreset = 'macro' | 'meso' | 'micro';

/** One car's row from the ACV ranker; rank 1 is the most likely faulty car. */
export interface AcvCarRank {
  rank: number;
  car: string;
  score: number | null;
  diagnosable: boolean;
  evidence: { feature: string; value: number | null }[];
}

export interface AcvResult {
  file_id: string;
  ranked_cars: string;
  ranked_cars_list: string[];
  most_likely_faulty_car: string;
  verdict: string;
  cars: AcvCarRank[];
}
export type SubsystemSelection = 'overview' | 'door' | 'acv' | 'shm' | 'rail';

/** Beginner hides jargon and leads with plain words; Expert shows raw engineering units. */
export type UiMode = 'beginner' | 'expert';
