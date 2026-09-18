"""
backend/models/inference_broker.py
===============================================================================
Multi-Model Inference Broker:
Coordinates concurrent subsystem model execution (Door, ACV, SHM, Rail Corrugation).
Aggregates subsystem anomalies into a holistic Fleet Health Index.
===============================================================================
"""

from backend.models.door_model import DoorPredictor
from backend.models.acv_model import ACVSubsystemModel
from backend.models.shm_model import SHMSubsystemModel
from backend.models.corrugation_model import RailCorrugationModel

class InferenceBroker:
    def __init__(self):
        self.door_model = DoorPredictor()
        self.acv_model = ACVSubsystemModel()
        self.shm_model = SHMSubsystemModel()
        self.rail_model = RailCorrugationModel()

    def process_subsystems(
        self,
        chainage_kp: float,
        speed_kmh: float,
        door_current: float = 8.2,
        door_transit_sec: float = 3.1,
        is_door_opening: bool = False,
        supply_temp: float = 19.2,
        return_temp: float = 26.5,
        compressor_kw: float = 4.8,
        acv_degraded: bool = False,
        rail_ground_override: bool = False,
        door_lubricated_override: bool = False
    ) -> dict:
        # 1. Rail Corrugation
        rail_data = self.rail_model.evaluate_chainage(chainage_kp, grounded_override=rail_ground_override)
        rail_severity = rail_data["severity_score"]

        # 2. Structural Health Monitoring (coupled with rail roughness)
        shm_data = self.shm_model.evaluate_vibration(
            speed_kmh=speed_kmh,
            corrugation_severity=rail_severity
        )

        # 3. Door Electromechanics
        effective_door_current = 8.2 if door_lubricated_override else door_current
        effective_transit_sec = 3.0 if door_lubricated_override else door_transit_sec
        door_data = self.door_model.evaluate_cycle(
            motor_current_amps=effective_door_current,
            nominal_current_amps=8.20,
            transit_time=effective_transit_sec,
            is_opening=is_door_opening
        )

        # 4. ACV Thermodynamics
        acv_data = self.acv_model.evaluate_telemetry(
            supply_temp=supply_temp,
            return_temp=return_temp,
            compressor_kw=compressor_kw,
            degraded=acv_degraded
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
