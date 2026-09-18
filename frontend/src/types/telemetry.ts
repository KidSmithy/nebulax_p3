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

export interface UnifiedTelemetryFrame {
  frame_id: number;
  timestamp: string;
  track_chainage_km: number;
  train_speed_kmh: number;
  fleet_health_index: number;
  subsystems: SubsystemsPayload;
}

export type CameraPreset = 'macro' | 'meso' | 'micro';
export type SubsystemSelection = 'overview' | 'door' | 'acv' | 'shm' | 'rail';
