import os
from pathlib import Path

# Paths
CORE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CORE_DIR.parent
ROOT_DIR = BACKEND_DIR.parent

# ---------------------------------------------------------------------------
# Environment / secrets
# Loaded from backend/.env. The API key is only ever read server-side; it is
# never included in any telemetry frame or REST response sent to the browser.
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv(BACKEND_DIR / ".env")
except ImportError:  # pragma: no cover - optional dependency
    pass


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
OPENAI_MODEL = (os.getenv("OPENAI_MODEL") or "gpt-5.6-terra").strip()
AI_INSIGHT_MIN_INTERVAL_SEC = _env_float("AI_INSIGHT_MIN_INTERVAL_SEC", 6.0)
AI_ENABLED = bool(OPENAI_API_KEY)

PS3_DIR = ROOT_DIR / "PS3"
DATASETS_DIR = PS3_DIR / "02_Datasets"

MODEL_DATA_DIR = BACKEND_DIR / "model_data"
DOOR_DIR = ROOT_DIR / "Door"
ACV_DIR = ROOT_DIR / "ACV"
RAIL_DIR = ROOT_DIR / "Rail_Corrugation"
SHM_DIR = ROOT_DIR / "SHM"

# Telemetry streaming configuration
STREAM_RATE_HZ = 10
STREAM_INTERVAL_SEC = 1.0 / STREAM_RATE_HZ

# Simulation and Track Geometry
DEFAULT_START_KP = 12.500   # Starting chainage in km
TRACK_MAX_KP = 35.000       # Corridor length in km
DEFAULT_SPEED_KMH = 68.4    # Nominal operating speed km/h

# Known Corrugation Alert Zones (Kilometre Posts)
CORRUGATION_ZONES = [
    {"kp_start": 14.800, "kp_end": 14.900, "wavelength_class": "SHORT_PITCH", "depth_microns": 42.0, "severity": 0.89, "urgency": "SCHEDULE_GRINDING_7D", "rank": 2},
    {"kp_start": 18.250, "kp_end": 18.350, "wavelength_class": "LONG_PITCH", "depth_microns": 36.5, "severity": 0.74, "urgency": "INSPECT_14D", "rank": 4},
    {"kp_start": 23.100, "kp_end": 23.220, "wavelength_class": "SHORT_PITCH", "depth_microns": 48.0, "severity": 0.94, "urgency": "IMMEDIATE_GRINDING_48H", "rank": 1},
]

# ---------------------------------------------------------------------------
# Metric interpretation thresholds.
# Single source of truth for "is this number good or bad?". Consumed by the AI
# insight service and mirrored by the frontend metric glossary so that the
# plain-language verdict shown to the operator always matches the backend.
# Each entry: (good_below, warn_below) -> above warn_below is critical.
# 'higher_is_better' metrics invert the comparison.
# ---------------------------------------------------------------------------
METRIC_THRESHOLDS = {
    # A healthy door peaks near 9.5 A mid-travel, so the healthy band must sit
    # above that peak or every normal cycle would be flagged.
    "door.motor_current_amps":      {"good": 10.0, "warn": 12.0, "unit": "A",  "higher_is_better": False},
    "door.anomaly_score":           {"good": 0.40, "warn": 0.65, "unit": "",   "higher_is_better": False},
    "door.transit_time_seconds":    {"good": 3.3,  "warn": 3.8,  "unit": "s",  "higher_is_better": False},
    "door.ghost_deviation_mm":      {"good": 5.0,  "warn": 15.0, "unit": "mm", "higher_is_better": False},
    "acv.delta_temp_c":             {"good": 7.5,  "warn": 6.0,  "unit": "\u00b0C", "higher_is_better": True},
    "acv.efficiency_rating":        {"good": 0.80, "warn": 0.65, "unit": "",   "higher_is_better": True},
    "acv.compressor_power_kw":      {"good": 5.0,  "warn": 5.8,  "unit": "kW", "higher_is_better": False},
    "acv.anomaly_score":            {"good": 0.35, "warn": 0.55, "unit": "",   "higher_is_better": False},
    "shm.vibration_rms_g":          {"good": 2.5,  "warn": 3.5,  "unit": "g",  "higher_is_better": False},
    "shm.bearing_defect_prob":      {"good": 0.20, "warn": 0.45, "unit": "",   "higher_is_better": False},
    "shm.fatigue_damage_index":     {"good": 0.40, "warn": 0.70, "unit": "",   "higher_is_better": False},
    "shm.anomaly_score":            {"good": 0.50, "warn": 0.70, "unit": "",   "higher_is_better": False},
    "rail_corrugation.depth_microns":  {"good": 20.0, "warn": 35.0, "unit": "\u03bcm", "higher_is_better": False},
    "rail_corrugation.severity_score": {"good": 0.35, "warn": 0.65, "unit": "",  "higher_is_better": False},
    "fleet_health_index":           {"good": 0.80, "warn": 0.60, "unit": "",   "higher_is_better": True},
}


def classify_metric(path: str, value) -> str:
    """Return 'GOOD' | 'WATCH' | 'ACTION_NEEDED' | 'UNKNOWN' for a metric value."""
    spec = METRIC_THRESHOLDS.get(path)
    if spec is None or value is None:
        return "UNKNOWN"
    good, warn = spec["good"], spec["warn"]
    if spec["higher_is_better"]:
        if value >= good:
            return "GOOD"
        return "WATCH" if value >= warn else "ACTION_NEEDED"
    if value <= good:
        return "GOOD"
    return "WATCH" if value <= warn else "ACTION_NEEDED"


# Maintenance interventions available to the What-If sandbox.
INTERVENTION_ACTIONS = [
    "ACTION_GRIND_RAIL",
    "ACTION_LUBRICATE_DOOR",
    "ACTION_REPLACE_FILTER",
    "ACTION_INSPECT_BEARING",
]
