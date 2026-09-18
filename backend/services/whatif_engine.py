"""
backend/services/whatif_engine.py
===============================================================================
Counterfactual Simulation Engine:
Evaluates operational maintenance interventions:
- Rail Grinding (removes short/long pitch corrugation, resets bogie shock loads)
- Door Guide Lubrication (eliminates motor overcurrent, restores nominal kinematics)
- ACV Filter Replacement (restores COP efficiency and nominal cooling delta)
- Axle-Box Bearing Overhaul (resets bearing defect probabilities)

The impact figures returned here are MEASURED, not asserted: the engine asks
the harmonizer to evaluate the identical physical instant with and without the
action and reports the difference. If an action ever stops having an effect,
that shows up honestly as a zero delta instead of being masked by a
hardcoded marketing number.
===============================================================================
"""

from typing import Any, Dict

from backend.core.config import INTERVENTION_ACTIONS
from backend.models.inference_broker import NO_INTERVENTIONS

# Operator-facing copy for each action, in plain language.
ACTION_CATALOG = {
    "ACTION_GRIND_RAIL": {
        "title": "Grind the rail smooth",
        "plain_description": (
            "A grinding train shaves the rippled top surface off the rail. "
            "Smoother rail means the wheels stop hammering the train."
        ),
        "technical_description": "Continuous track grinding across corrugated KP chainage",
        "affects": ["rail_corrugation", "shm"],
        "location_sensitive": True,
    },
    "ACTION_LUBRICATE_DOOR": {
        "title": "Grease the door tracks",
        "plain_description": (
            "Clean and lubricate the rails the door slides along, so the motor "
            "no longer has to fight friction to move the doors."
        ),
        "technical_description": "Clean guide-rails & apply low-viscosity PTFE lubricant",
        "affects": ["door"],
        "location_sensitive": False,
    },
    "ACTION_REPLACE_FILTER": {
        "title": "Replace the aircon filter",
        "plain_description": (
            "Fit a clean air filter and top up the refrigerant, so the aircon "
            "can push cold air through freely again."
        ),
        "technical_description": "Fresh filter pack & refrigerant charge replenishment",
        "affects": ["acv"],
        "location_sensitive": False,
    },
    "ACTION_INSPECT_BEARING": {
        "title": "Service the wheel bearing",
        "plain_description": (
            "Ultrasonically check the axle bearing for cracks and re-grease it, "
            "removing the risk of a seized wheel."
        ),
        "technical_description": "Ultrasonic flaw evaluation & high-pressure re-greasing",
        "affects": ["shm"],
        "location_sensitive": False,
    },
}


class WhatIfEngine:
    def __init__(self, harmonizer=None):
        self.harmonizer = harmonizer
        self.active_interventions = dict(NO_INTERVENTIONS)

    def simulate_action(self, action_type: str, enabled: bool = True) -> Dict[str, Any]:
        if action_type not in INTERVENTION_ACTIONS:
            return {
                "action": action_type,
                "status": "UNKNOWN_ACTION",
                "known_actions": INTERVENTION_ACTIONS,
            }

        self.active_interventions[action_type] = bool(enabled)

        applied = False
        if self.harmonizer:
            applied = self.harmonizer.apply_action(action_type, enabled)
            self.active_interventions = dict(self.harmonizer.interventions)

        catalog = ACTION_CATALOG.get(action_type, {})
        result = {
            "action": action_type,
            "enabled": enabled,
            "title": catalog.get("title"),
            "plain_description": catalog.get("plain_description"),
            "technical_description": catalog.get("technical_description"),
            "active_interventions": dict(self.active_interventions),
            "status": "APPLIED" if enabled else "REVERTED",
            "wired_to_simulation": applied,
        }

        # Measure what this action actually did, at this instant, right now.
        if self.harmonizer:
            result["measured_impact"] = self.measure_action(action_type)
            result["next_corrugation_zone"] = self.harmonizer.next_corrugation_zone(
                self.harmonizer.generator.current_kp
            )

        return result

    def measure_action(self, action_type: str) -> Dict[str, Any]:
        """
        Isolate one action: evaluate the current instant with only that action
        toggled versus the current baseline, and report the measured deltas.
        """
        h = self.harmonizer
        if h is None:
            return {}

        raw = h.generator.step(dt=0.0)
        seed = raw["frame_id"]

        # A parked door cannot demonstrate the benefit of lubrication, so the
        # door action is measured at a representative point in its travel.
        measured_at_peak_travel = False
        if action_type == "ACTION_LUBRICATE_DOOR" and not raw.get("door_is_moving", True):
            raw = h.generator.peak_travel_snapshot(raw)
            measured_at_peak_travel = True

        without = {**h.interventions, action_type: False}
        with_it = {**h.interventions, action_type: True}

        base = h.broker.process_subsystems(raw, without, seed=seed)
        after = h.broker.process_subsystems(raw, with_it, seed=seed)

        from backend.core.harmonizer import COMPARED_METRICS, _dig

        base_flat, after_flat = h._flatten(base), h._flatten(after)
        deltas = []
        for path, label, unit, decimals, lower_is_better in COMPARED_METRICS:
            b, a = _dig(base_flat, path), _dig(after_flat, path)
            if b is None or a is None:
                continue
            delta = round(float(a) - float(b), 4)
            if abs(delta) < 1e-6:
                continue
            deltas.append({
                "path": path,
                "label": label,
                "unit": unit,
                "decimals": decimals,
                "before": round(float(b), 4),
                "after": round(float(a), 4),
                "delta": delta,
                "direction": "better" if (delta < 0) == lower_is_better else "worse",
            })

        has_effect = bool(deltas)
        note = None
        if measured_at_peak_travel:
            note = (
                "The door is parked right now, so its motor is idle. These figures are "
                "measured at the busiest point of a door cycle, which is where the "
                "improvement actually shows."
            )
        elif not has_effect and ACTION_CATALOG.get(action_type, {}).get("location_sensitive"):
            zone = h.next_corrugation_zone(raw["track_chainage_km"])
            note = (
                f"No effect here because the track at KP {raw['track_chainage_km']:.3f} "
                f"is already smooth. The next rough section starts at "
                f"KP {zone['kp_start']:.3f}."
            )
        elif not has_effect:
            note = "This subsystem is already in nominal condition, so there is nothing to repair."

        return {
            "has_measurable_effect": has_effect,
            "changed_metrics": deltas,
            "note": note,
            "measured_at_peak_travel": measured_at_peak_travel,
        }
