"""
backend/utils/mock_generator.py
===============================================================================
Dynamic Telemetry Stream Generator:
Synthesizes continuous, physically coupled train operations:
- Kinematic progression along track chainage KP(t)
- Speed profiles with cruising, acceleration, and deceleration
- Periodic door cycle activations (Opening, Closing, Station Dwells)
- ACV thermodynamic cycles with compressor heat loads
- Corrugation-induced vibration spikes
===============================================================================
"""

import math
import time
import numpy as np
from datetime import datetime, timezone
from backend.core.config import DEFAULT_START_KP, DEFAULT_SPEED_KMH, TRACK_MAX_KP

class TelemetryGenerator:
    def __init__(self, start_kp: float = DEFAULT_START_KP, nominal_speed: float = DEFAULT_SPEED_KMH):
        self.current_kp = start_kp
        self.speed_kmh = nominal_speed
        self.frame_id = 1000
        self.start_time = time.time()
        self.elapsed_sec = 0.0

        # Door state machine
        self.door_cycle_timer = 0.0
        self.is_door_cycle_active = False
        self.door_cycle_phase = "IDLE" # IDLE, OPENING, DWELL, CLOSING
        self.door_phase_progress = 0.0
        self.door_current = 0.0
        self.door_transit_time = 3.2

        # ACV state
        self.acv_compressor_power = 4.82
        self.supply_temp = 19.2
        self.return_temp = 26.5
        self.acv_degraded = False

        # Overrides (Counterfactuals)
        self.rail_ground_override = False
        self.door_lubricated_override = False

    def step(self, dt: float = 0.1, speed_multiplier: float = 1.0) -> dict:
        """
        Advances the physical simulation by dt seconds (scaled by speed_multiplier).
        """
        effective_dt = dt * speed_multiplier
        self.elapsed_sec += effective_dt
        self.frame_id += 1

        # 1. Update Speed with realistic gentle cruising fluctuation
        speed_var = 2.5 * math.sin(self.elapsed_sec * 0.05) + 1.2 * math.cos(self.elapsed_sec * 0.13)
        self.speed_kmh = float(np.clip(DEFAULT_SPEED_KMH + speed_var, 45.0, 85.0))

        # 2. Integrate continuous chainage KP(t) = KP0 + integral v(tau) dtau
        # v (km/h) / 3600 = km/s
        delta_kp = (self.speed_kmh / 3600.0) * effective_dt
        self.current_kp += delta_kp
        if self.current_kp > TRACK_MAX_KP:
            self.current_kp = DEFAULT_START_KP

        # 3. Advance Door State Machine (Cycle every 25 seconds of simulated time)
        self.door_cycle_timer += effective_dt
        cycle_period = 25.0
        cycle_pos = self.door_cycle_timer % cycle_period

        if cycle_pos < 3.5:
            self.door_cycle_phase = "OPENING"
            progress = cycle_pos / 3.5
            # Motor current bell curve with initial breakaway spike
            self.door_current = float(np.clip(11.2 * math.sin(progress * math.pi) + 2.0, 1.0, 12.5))
            self.door_transit_time = 3.5
        elif cycle_pos < 8.0:
            self.door_cycle_phase = "DWELL_OPEN"
            self.door_current = 0.4
            self.door_transit_time = 0.0
        elif cycle_pos < 11.5:
            self.door_cycle_phase = "CLOSING"
            progress = (cycle_pos - 8.0) / 3.5
            self.door_current = float(np.clip(11.85 * math.sin(progress * math.pi) + 1.8, 1.0, 13.0))
            self.door_transit_time = 3.42
        else:
            self.door_cycle_phase = "CLOSED_LOCKED"
            self.door_current = 0.0
            self.door_transit_time = 0.0

        # 4. ACV thermodynamics progression
        ambient_drift = 0.4 * math.sin(self.elapsed_sec * 0.02)
        self.return_temp = 26.5 + ambient_drift
        self.supply_temp = 19.2 + 0.3 * math.cos(self.elapsed_sec * 0.03)
        self.acv_compressor_power = 4.82 + 0.35 * math.sin(self.elapsed_sec * 0.08)

        iso_timestamp = datetime.now(timezone.utc).isoformat()

        return {
            "frame_id": self.frame_id,
            "timestamp": iso_timestamp,
            "track_chainage_km": round(self.current_kp, 3),
            "train_speed_kmh": round(self.speed_kmh, 1),
            "door_current": self.door_current,
            "door_transit_time": self.door_transit_time,
            "is_door_opening": (self.door_cycle_phase == "OPENING"),
            "door_phase": self.door_cycle_phase,
            "supply_temp": self.supply_temp,
            "return_temp": self.return_temp,
            "compressor_kw": self.acv_compressor_power,
            "acv_degraded": self.acv_degraded,
            "rail_ground_override": self.rail_ground_override,
            "door_lubricated_override": self.door_lubricated_override
        }

    def seek(self, target_kp: float):
        self.current_kp = max(DEFAULT_START_KP, min(target_kp, TRACK_MAX_KP))
