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

The generator emits PURE BASELINE physics: the untouched condition of the
asset. Maintenance interventions are NOT applied here - they are applied
downstream in the inference broker. That separation is what allows the
harmonizer to evaluate the same physical instant twice (once with
interventions, once without) to produce a true counterfactual comparison.
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
        self.door_cycle_phase = "IDLE"  # IDLE, OPENING, DWELL_OPEN, CLOSING, CLOSED_LOCKED
        self.door_phase_progress = 0.0
        self.door_current = 0.0
        self.door_transit_time = 3.2

        # ACV state
        self.acv_compressor_power = 4.82
        self.supply_temp = 19.2
        self.return_temp = 26.5

        # ------------------------------------------------------------------
        # Baseline asset condition (the faults that ALREADY exist on this
        # train before any maintenance is carried out). Without these, the
        # What-If interventions would have nothing to repair and would appear
        # to do nothing when toggled.
        # ------------------------------------------------------------------
        # Door guide-rail friction: raises motor current above the nominal
        # envelope and stretches the transit time.
        self.door_friction_factor = 1.32
        # ACV fresh-air filter is partially clogged, suppressing the COP.
        self.acv_filter_clogged = True
        # Incipient axle-box bearing inner-race defect. This is INDEPENDENT of
        # track roughness - real bearing defects show up as discrete defect
        # frequencies, not merely as broadband RMS.
        self.bearing_wear = 0.42

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

        # Healthy mechanical envelope, before guide-rail friction is applied.
        if cycle_pos < 3.5:
            self.door_cycle_phase = "OPENING"
            progress = cycle_pos / 3.5
            nominal_current = 7.9 * math.sin(progress * math.pi) + 1.4
            nominal_transit = 3.05
        elif cycle_pos < 8.0:
            self.door_cycle_phase = "DWELL_OPEN"
            progress = 0.0
            nominal_current = 0.4
            nominal_transit = 0.0
        elif cycle_pos < 11.5:
            self.door_cycle_phase = "CLOSING"
            progress = (cycle_pos - 8.0) / 3.5
            nominal_current = 8.2 * math.sin(progress * math.pi) + 1.3
            nominal_transit = 3.10
        else:
            self.door_cycle_phase = "CLOSED_LOCKED"
            progress = 0.0
            nominal_current = 0.0
            nominal_transit = 0.0

        self.door_phase_progress = progress
        self.door_nominal_current = float(round(nominal_current, 3))
        self.door_nominal_transit = nominal_transit

        # Friction only loads the motor while the leaves are actually moving.
        is_moving = self.door_cycle_phase in ("OPENING", "CLOSING")
        if is_moving:
            self.door_current = float(np.clip(nominal_current * self.door_friction_factor, 0.0, 14.0))
            self.door_transit_time = round(nominal_transit * (1.0 + (self.door_friction_factor - 1.0) * 0.30), 2)
        else:
            self.door_current = float(nominal_current)
            self.door_transit_time = nominal_transit

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
            "door_nominal_current": self.door_nominal_current,
            "door_transit_time": self.door_transit_time,
            "door_nominal_transit": self.door_nominal_transit,
            "door_is_moving": is_moving,
            "is_door_opening": (self.door_cycle_phase == "OPENING"),
            "door_phase": self.door_cycle_phase,
            "door_friction_factor": self.door_friction_factor,
            "supply_temp": self.supply_temp,
            "return_temp": self.return_temp,
            "compressor_kw": self.acv_compressor_power,
            "acv_filter_clogged": self.acv_filter_clogged,
            "bearing_wear": self.bearing_wear,
        }

    def seek(self, target_kp: float):
        self.current_kp = max(DEFAULT_START_KP, min(target_kp, TRACK_MAX_KP))

    def peak_travel_snapshot(self, raw: dict) -> dict:
        """
        A copy of `raw` repositioned to the busiest instant of a door cycle.

        A parked door draws no current, so there is nothing to diagnose and
        nothing for lubrication to improve. Evaluating the door action against
        the current instant would therefore report "no effect" for the ~18 of
        every 25 seconds the door spends closed, which reads as a broken
        feature. This provides a representative mid-travel instant instead, so
        the benefit can always be quantified. Callers must label the result as
        being measured at peak travel rather than right now.
        """
        snap = dict(raw)
        # Mid-travel of a CLOSING cycle: sin(pi/2) = 1, the current peak.
        nominal_peak = 8.2 + 1.3
        snap["door_nominal_current"] = round(nominal_peak, 3)
        snap["door_nominal_transit"] = 3.10
        snap["door_current"] = float(np.clip(nominal_peak * self.door_friction_factor, 0.0, 14.0))
        snap["door_transit_time"] = round(
            3.10 * (1.0 + (self.door_friction_factor - 1.0) * 0.30), 2
        )
        snap["door_is_moving"] = True
        snap["is_door_opening"] = False
        snap["door_phase"] = "CLOSING"
        return snap
