"""
backend/models/inference_broker.py
===============================================================================
Multi-Model Inference Broker:
Coordinates concurrent subsystem model execution (Door, ACV, SHM, Rail Corrugation).
Aggregates subsystem anomalies into a holistic Fleet Health Index.

This is the single place where maintenance interventions are translated into
physical effects. Every action in INTERVENTION_ACTIONS must have an observable
consequence here, otherwise it is a dead toggle in the UI.
===============================================================================
"""

import numpy as np

from backend.models.door_model import DoorPredictor
from backend.models.acv_model import ACVSubsystemModel
from backend.models.shm_model import SHMSubsystemModel
from backend.models.corrugation_model import RailCorrugationModel

NO_INTERVENTIONS = {
    "ACTION_GRIND_RAIL": False,
    "ACTION_LUBRICATE_DOOR": False,
    "ACTION_REPLACE_FILTER": False,
    "ACTION_INSPECT_BEARING": False,
}


class InferenceBroker:
    def __init__(self):
        self.door_model = DoorPredictor()
        self.acv_model = ACVSubsystemModel()
        self.shm_model = SHMSubsystemModel()
        self.rail_model = RailCorrugationModel()

    def process_subsystems(self, raw: dict, interventions: dict = None, seed: int = None) -> dict:
        """
        Runs all four subsystem models for one physical instant.

        raw:            the untouched physical state from TelemetryGenerator.step()
        interventions:  which maintenance actions are currently applied
        seed:           fixes the sampling noise so that the same instant can be
                        evaluated twice (with and without interventions) and the
                        difference attributed purely to the maintenance action.
        """
        iv = {**NO_INTERVENTIONS, **(interventions or {})}
        rng = np.random.default_rng(seed)

        # ------------------------------------------------------------------
        # 1. Rail Corrugation -- grinding removes the surface roughness.
        # ------------------------------------------------------------------
        rail_data = self.rail_model.evaluate_chainage(
            raw["track_chainage_km"],
            grounded_override=iv["ACTION_GRIND_RAIL"],
            rng=rng,
        )
        rail_severity = rail_data["severity_score"]

        # ------------------------------------------------------------------
        # 2. SHM -- coupled to rail roughness AND to the bearing's own state.
        #    Re-greasing/servicing the bearing collapses its wear term, which
        #    is why ACTION_INSPECT_BEARING now has a visible effect even on
        #    perfectly smooth track.
        # ------------------------------------------------------------------
        bearing_wear = 0.04 if iv["ACTION_INSPECT_BEARING"] else raw.get("bearing_wear", 0.0)
        shm_data = self.shm_model.evaluate_vibration(
            speed_kmh=raw["train_speed_kmh"],
            corrugation_severity=rail_severity,
            bearing_wear=bearing_wear,
            rng=rng,
        )

        # ------------------------------------------------------------------
        # 3. Door -- lubrication REMOVES the guide-rail friction multiplier.
        #    It must not pin the current to a constant: the motor still draws
        #    its normal bell-curve while moving and ~0 A while parked.
        # ------------------------------------------------------------------
        nominal_current = raw.get("door_nominal_current", 8.2)
        nominal_transit = raw.get("door_nominal_transit", 3.05)
        is_moving = raw.get("door_is_moving", True)

        if iv["ACTION_LUBRICATE_DOOR"]:
            # Lubricated: friction term removed, only a small residual remains.
            residual = 1.0 + (raw.get("door_friction_factor", 1.0) - 1.0) * 0.08
            effective_current = nominal_current * (residual if is_moving else 1.0)
            effective_transit = round(nominal_transit * (1.0 + (residual - 1.0) * 0.30), 2)
        else:
            effective_current = raw["door_current"]
            effective_transit = raw["door_transit_time"]

        door_data = self.door_model.evaluate_cycle(
            motor_current_amps=effective_current,
            nominal_current_amps=nominal_current,
            transit_time=effective_transit,
            is_opening=raw.get("is_door_opening", False),
            is_moving=is_moving,
            nominal_transit_time=nominal_transit,
            rng=rng,
        )

        # ------------------------------------------------------------------
        # 4. ACV -- replacing the filter clears the clog. Note the polarity:
        #    action APPLIED means the filter is FRESH (not degraded).
        # ------------------------------------------------------------------
        acv_degraded = raw.get("acv_filter_clogged", False) and not iv["ACTION_REPLACE_FILTER"]
        # A fresh filter restores airflow, so heat exchange improves and the
        # compressor no longer has to work against a restricted duct.
        supply_temp = raw["supply_temp"] - (1.9 if not acv_degraded else 0.0)
        compressor_kw = raw["compressor_kw"] - (0.62 if not acv_degraded else 0.0)

        acv_data = self.acv_model.evaluate_telemetry(
            supply_temp=supply_temp,
            return_temp=raw["return_temp"],
            compressor_kw=compressor_kw,
            degraded=acv_degraded,
        )

        # Composite Fleet Health Index (1.0 = optimal, lower = elevated risk)
        penalty = (
            0.25 * door_data["anomaly_score"] +
            0.25 * acv_data["anomaly_score"] +
            0.30 * shm_data["anomaly_score"] +
            0.20 * rail_severity
        )
        fleet_health_index = round(float(max(0.10, 1.0 - penalty)), 2)

        return {
            "fleet_health_index": fleet_health_index,
            "door": door_data,
            "acv": acv_data,
            "shm": shm_data,
            "rail_corrugation": rail_data
        }
