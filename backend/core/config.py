import os
from pathlib import Path

# Paths
CORE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CORE_DIR.parent
ROOT_DIR = BACKEND_DIR.parent

PS3_DIR = ROOT_DIR / "PS3"
DATASETS_DIR = PS3_DIR / "02_Datasets"

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
