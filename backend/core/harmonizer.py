"""
backend/core/harmonizer.py
===============================================================================
Harmonization & Telemetry Frame Builder:
Harmonizes multi-rate domain signals into unified 10 Hz telemetry frames
strictly conforming to the digital twin specification schema.

Every frame is evaluated TWICE from the same physical instant using the same
noise seed:
  * "current"  - with the operator's maintenance interventions applied
  * "baseline" - with every intervention reverted

The difference between the two is a genuine measured counterfactual rather
than a hardcoded marketing figure, and it is shipped inside the frame so the
UI can show the operator exactly what their action changed.
===============================================================================
"""

from backend.core.config import CORRUGATION_ZONES, INTERVENTION_ACTIONS, classify_metric
from backend.models.inference_broker import InferenceBroker, NO_INTERVENTIONS
from backend.utils.mock_generator import TelemetryGenerator

# Metrics tracked in the counterfactual comparison.
# (frame path, label, unit, decimals, lower_is_better)
COMPARED_METRICS = [
    ("fleet_health_index", "Overall health", "%", 0, False),
    ("door.motor_current_amps", "Door motor current", "A", 2, True),
    ("door.anomaly_score", "Door fault risk", "%", 0, True),
    ("door.transit_time_seconds", "Door travel time", "s", 2, True),
    ("door.ghost_deviation_mm", "Door timing lag", "mm", 1, True),
    ("acv.delta_temp_c", "Cooling performance", "\u00b0C", 1, False),
    ("acv.efficiency_rating", "Aircon efficiency", "%", 0, False),
    ("acv.compressor_power_kw", "Compressor power", "kW", 2, True),
    ("shm.vibration_rms_g", "Bogie vibration", "g", 2, True),
    ("shm.bearing_defect_prob", "Bearing failure risk", "%", 0, True),
    ("shm.fatigue_damage_index", "Metal fatigue", "%", 0, True),
    ("rail_corrugation.depth_microns", "Rail roughness", "\u03bcm", 1, True),
    ("rail_corrugation.severity_score", "Track severity", "%", 0, True),
]


def _dig(frame_like: dict, path: str):
    """Fetch a dotted path out of a frame's flattened subsystem dict."""
    parts = path.split(".")
    if len(parts) == 1:
        return frame_like.get(parts[0])
    node = frame_like.get(parts[0], {})
    return node.get(parts[1]) if isinstance(node, dict) else None


class TelemetryHarmonizer:
    def __init__(self):
        self.generator = TelemetryGenerator()
        self.broker = InferenceBroker()
        self.interventions = dict(NO_INTERVENTIONS)

    # ------------------------------------------------------------------
    # Intervention state
    # ------------------------------------------------------------------
    def apply_action(self, action_name: str, enabled: bool = True) -> bool:
        """
        Records a maintenance intervention. Returns True if the action is known.

        Polarity is uniform for every action: enabled=True means the
        maintenance HAS BEEN CARRIED OUT and the asset should improve.
        """
        if action_name not in self.interventions:
            return False
        self.interventions[action_name] = bool(enabled)
        return True

    def reset_actions(self):
        self.interventions = dict(NO_INTERVENTIONS)

    @property
    def active_actions(self):
        return [a for a in INTERVENTION_ACTIONS if self.interventions.get(a)]

    # ------------------------------------------------------------------
    # Frame construction
    # ------------------------------------------------------------------
    def generate_next_frame(self, dt: float = 0.1, speed_multiplier: float = 1.0) -> dict:
        # Step physical generator once - this is the shared physical instant.
        raw = self.generator.step(dt=dt, speed_multiplier=speed_multiplier)
        seed = raw["frame_id"]

        # Pass 1: the world as it is now, with the operator's actions applied.
        current = self.broker.process_subsystems(raw, self.interventions, seed=seed)

        # Overwrite cycle state with high-level phase if idle/locked
        if raw["door_phase"] in ["DWELL_OPEN", "CLOSED_LOCKED"]:
            current["door"]["cycle_state"] = raw["door_phase"]

        frame_flat = self._flatten(current)

        frame = {
            "frame_id": raw["frame_id"],
            "timestamp": raw["timestamp"],
            "track_chainage_km": raw["track_chainage_km"],
            "train_speed_kmh": raw["train_speed_kmh"],
            "fleet_health_index": current["fleet_health_index"],
            "subsystems": {
                "door": current["door"],
                "acv": current["acv"],
                "shm": current["shm"],
                "rail_corrugation": current["rail_corrugation"],
            },
            "active_interventions": dict(self.interventions),
            "plain_status": self._plain_status(frame_flat),
            "next_corrugation_zone": self.next_corrugation_zone(raw["track_chainage_km"]),
        }

        # Pass 2: the same instant with every intervention reverted, so the
        # delta is attributable solely to the maintenance actions.
        active = self.active_actions
        if active:
            baseline = self.broker.process_subsystems(raw, NO_INTERVENTIONS, seed=seed)
            frame["counterfactual"] = self._counterfactual(
                baseline_flat=self._flatten(baseline),
                current_flat=frame_flat,
                active_actions=active,
            )
        else:
            frame["counterfactual"] = {
                "active": False,
                "active_actions": [],
                "metrics": [],
                "headline": None,
            }

        return frame

    # ------------------------------------------------------------------
    @staticmethod
    def _flatten(result: dict) -> dict:
        return {
            "fleet_health_index": result["fleet_health_index"],
            "door": result["door"],
            "acv": result["acv"],
            "shm": result["shm"],
            "rail_corrugation": result["rail_corrugation"],
        }

    @staticmethod
    def _plain_status(flat: dict) -> dict:
        """Per-metric GOOD / WATCH / ACTION_NEEDED verdicts for the UI."""
        out = {}
        for path, *_ in COMPARED_METRICS:
            value = _dig(flat, path)
            if value is not None:
                out[path] = classify_metric(path, value)
        return out

    @staticmethod
    def _counterfactual(baseline_flat: dict, current_flat: dict, active_actions: list) -> dict:
        metrics = []
        improved = 0
        for path, label, unit, decimals, lower_is_better in COMPARED_METRICS:
            before = _dig(baseline_flat, path)
            after = _dig(current_flat, path)
            if before is None or after is None:
                continue
            delta = round(float(after) - float(before), 4)
            if abs(delta) < 1e-6:
                direction = "unchanged"
            elif (delta < 0) == lower_is_better:
                direction = "better"
                improved += 1
            else:
                direction = "worse"

            pct = None
            if before not in (0, None) and abs(before) > 1e-9:
                pct = round(delta / abs(float(before)) * 100.0, 1)

            metrics.append({
                "path": path,
                "label": label,
                "unit": unit,
                "decimals": decimals,
                "before": round(float(before), 4),
                "after": round(float(after), 4),
                "delta": delta,
                "percent_change": pct,
                "direction": direction,
                "status_before": classify_metric(path, before),
                "status_after": classify_metric(path, after),
            })

        health = next((m for m in metrics if m["path"] == "fleet_health_index"), None)
        if health and health["direction"] == "better":
            headline = (
                f"Overall health improved from {health['before'] * 100:.0f}% "
                f"to {health['after'] * 100:.0f}%"
            )
        elif health and health["direction"] == "unchanged":
            headline = "No measurable change at this location yet"
        elif health:
            headline = (
                f"Overall health moved from {health['before'] * 100:.0f}% "
                f"to {health['after'] * 100:.0f}%"
            )
        else:
            headline = None

        return {
            "active": True,
            "active_actions": active_actions,
            "metrics": metrics,
            "improved_count": improved,
            "headline": headline,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def next_corrugation_zone(current_kp: float):
        """
        The nearest rough section ahead of the train.

        Rail grinding only has a large effect inside a corrugated zone, and the
        default start (KP 12.500) sits far from every zone. Surfacing this lets
        the UI offer to jump the operator somewhere the action is meaningful,
        instead of leaving them to conclude the feature is broken.
        """
        ahead = [z for z in CORRUGATION_ZONES if z["kp_start"] > current_kp]
        zone = min(ahead, key=lambda z: z["kp_start"]) if ahead else min(
            CORRUGATION_ZONES, key=lambda z: z["kp_start"]
        )
        mid = round((zone["kp_start"] + zone["kp_end"]) / 2.0, 3)
        inside = zone["kp_start"] - 0.05 <= current_kp <= zone["kp_end"] + 0.05
        return {
            "kp_start": zone["kp_start"],
            "kp_end": zone["kp_end"],
            "kp_centre": mid,
            "severity": zone["severity"],
            "wavelength_class": zone["wavelength_class"],
            "urgency": zone["urgency"],
            "distance_km": round(zone["kp_start"] - current_kp, 3),
            "is_inside_zone": inside,
        }

    def seek_chainage(self, target_kp: float):
        self.generator.seek(target_kp)
