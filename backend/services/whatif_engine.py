"""
backend/services/whatif_engine.py
===============================================================================
Counterfactual Simulation Engine:
Evaluates operational maintenance interventions:
- Rail Grinding (removes short/long pitch corrugation, resets bogie shock loads)
- Door Guide Lubrication (eliminates motor overcurrent, restores nominal kinematics)
- ACV Filter Replacement (restores COP efficiency and nominal cooling delta)
- Axle-Box Bearing Overhaul (resets bearing defect probabilities)
===============================================================================
"""

from typing import Dict, Any

class WhatIfEngine:
    def __init__(self, harmonizer=None):
        self.harmonizer = harmonizer
        self.active_interventions = {
            "ACTION_GRIND_RAIL": False,
            "ACTION_LUBRICATE_DOOR": False,
            "ACTION_REPLACE_FILTER": False,
            "ACTION_INSPECT_BEARING": False
        }

    def simulate_action(self, action_type: str, enabled: bool = True) -> Dict[str, Any]:
        if action_type in self.active_interventions:
            self.active_interventions[action_type] = enabled

        # Inform harmonizer if connected
        if self.harmonizer:
            self.harmonizer.apply_action(action_type, enabled)

        # Baseline impact calculation
        impacts = {
            "ACTION_GRIND_RAIL": {
                "description": "Autonomous track grinding along high-roughness kilometre posts",
                "corrugation_depth_delta": -33.5 if enabled else 0.0,
                "vibration_rms_reduction_g": -2.1 if enabled else 0.0,
                "projected_weld_life_extension_pct": +45.0 if enabled else 0.0,
                "health_index_gain": +0.18 if enabled else 0.0
            },
            "ACTION_LUBRICATE_DOOR": {
                "description": "Electromechanical guide-rail lubrication & track cleaning",
                "motor_current_peak_delta_a": -3.65 if enabled else 0.0,
                "ghost_delay_elimination_mm": -18.2 if enabled else 0.0,
                "cycle_duration_delta_sec": -0.42 if enabled else 0.0,
                "health_index_gain": +0.12 if enabled else 0.0
            },
            "ACTION_REPLACE_FILTER": {
                "description": "Cabin fresh-air intake filter renewal & refrigerant recharge",
                "thermal_delta_recovery_c": +2.2 if enabled else 0.0,
                "compressor_power_reduction_kw": -0.75 if enabled else 0.0,
                "health_index_gain": +0.08 if enabled else 0.0
            },
            "ACTION_INSPECT_BEARING": {
                "description": "Ultrasonic bogie bearing diagnostic & re-greasing",
                "bearing_defect_probability_reduction": -0.65 if enabled else 0.0,
                "health_index_gain": +0.10 if enabled else 0.0
            }
        }

        return {
            "action": action_type,
            "enabled": enabled,
            "active_interventions": self.active_interventions,
            "projected_impact": impacts.get(action_type, {}),
            "status": "APPLIED" if enabled else "REVERTED"
        }
