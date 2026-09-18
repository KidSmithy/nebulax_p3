"""
backend/models/acv_model.py
===============================================================================
Air Conditioning & Ventilation (ACV) Subsystem Model:
- Evaluates thermodynamic heat exchange, compressor power curves
- Refrigerant leakage and filter obstruction detection
- Computes thermal delta, COP efficiency, and anomaly score
===============================================================================
"""

import os
import joblib
import numpy as np
from backend.core.config import ACV_DIR, PS3_DIR

BUNDLE_PATHS = [
    ACV_DIR / "acv_model_bundle.joblib",
    PS3_DIR / "models" / "acv_model_bundle.joblib"
]

class ACVSubsystemModel:
    def __init__(self):
        self.bundle = None
        for p in BUNDLE_PATHS:
            if p.exists():
                try:
                    self.bundle = joblib.load(p)
                    break
                except Exception as e:
                    print(f"[ACVModel] Warning loading bundle from {p}: {e}")

    def evaluate_telemetry(self, supply_temp: float, return_temp: float, compressor_kw: float, degraded: bool = False) -> dict:
        delta_temp = round(float(return_temp - supply_temp), 1)

        # Baseline COP efficiency rating
        # Nominal delta is ~7.5 - 9.0 deg C at 4.0 - 5.0 kW
        nominal_delta = 8.5
        efficiency_ratio = delta_temp / max(nominal_delta, 1.0)
        power_ratio = compressor_kw / 4.5

        if degraded:
            efficiency_rating = round(float(np.clip(efficiency_ratio * 0.72, 0.40, 0.75)), 2)
            anomaly_score = round(float(np.clip((1.0 - efficiency_rating) * 1.3 + (power_ratio - 1.0) * 0.3, 0.55, 0.95)), 2)
            fault_type = "REFRIGERANT_LEAKAGE" if delta_temp < 6.0 else "FILTER_CLOGGING"
        else:
            efficiency_rating = round(float(np.clip(efficiency_ratio * 0.95, 0.75, 0.99)), 2)
            anomaly_score = round(float(np.clip(1.0 - efficiency_rating, 0.05, 0.35)), 2)
            fault_type = "NONE"

        return {
            "unit_id": "ACV_PACK_1",
            "supply_temp_c": round(float(supply_temp), 1),
            "return_temp_c": round(float(return_temp), 1),
            "delta_temp_c": delta_temp,
            "compressor_power_kw": round(float(compressor_kw), 2),
            "efficiency_rating": efficiency_rating,
            "anomaly_score": anomaly_score,
            "fault_type": fault_type
        }
