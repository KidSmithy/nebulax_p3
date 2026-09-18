"""
backend/core/harmonizer.py
===============================================================================
Harmonization & Telemetry Frame Builder:
Harmonizes multi-rate domain signals into unified 10 Hz telemetry frames
strictly conforming to the digital twin specification schema.
===============================================================================
"""

from backend.models.inference_broker import InferenceBroker
from backend.utils.mock_generator import TelemetryGenerator

class TelemetryHarmonizer:
    def __init__(self):
        self.generator = TelemetryGenerator()
        self.broker = InferenceBroker()

    def generate_next_frame(self, dt: float = 0.1, speed_multiplier: float = 1.0) -> dict:
        # Step physical generator
        raw = self.generator.step(dt=dt, speed_multiplier=speed_multiplier)

        # Run multi-model inference pipeline
        subsystems_data = self.broker.process_subsystems(
            chainage_kp=raw["track_chainage_km"],
            speed_kmh=raw["train_speed_kmh"],
            door_current=raw["door_current"],
            door_transit_sec=raw["door_transit_time"],
            is_door_opening=raw["is_door_opening"],
            supply_temp=raw["supply_temp"],
            return_temp=raw["return_temp"],
            compressor_kw=raw["compressor_kw"],
            acv_degraded=raw["acv_degraded"],
            rail_ground_override=raw["rail_ground_override"],
            door_lubricated_override=raw["door_lubricated_override"]
        )

        # Overwrite cycle state with high-level phase if idle/locked
        if raw["door_phase"] in ["DWELL_OPEN", "CLOSED_LOCKED"]:
            subsystems_data["door"]["cycle_state"] = raw["door_phase"]

        # Build standardized schema
        frame = {
            "frame_id": raw["frame_id"],
            "timestamp": raw["timestamp"],
            "track_chainage_km": raw["track_chainage_km"],
            "train_speed_kmh": raw["train_speed_kmh"],
            "fleet_health_index": subsystems_data["fleet_health_index"],
            "subsystems": {
                "door": subsystems_data["door"],
                "acv": subsystems_data["acv"],
                "shm": subsystems_data["shm"],
                "rail_corrugation": subsystems_data["rail_corrugation"]
            }
        }
        return frame

    def seek_chainage(self, target_kp: float):
        self.generator.seek(target_kp)

    def apply_action(self, action_name: str, enabled: bool = True):
        if action_name == "ACTION_GRIND_RAIL":
            self.generator.rail_ground_override = enabled
        elif action_name == "ACTION_LUBRICATE_DOOR":
            self.generator.door_lubricated_override = enabled
        elif action_name == "ACTION_REPLACE_FILTER":
            self.generator.acv_degraded = not enabled
