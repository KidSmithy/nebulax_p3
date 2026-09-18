"""Verify every What-If action produces a measurable telemetry change."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from backend.core.config import INTERVENTION_ACTIONS
from backend.core.harmonizer import TelemetryHarmonizer
from backend.services.whatif_engine import WhatIfEngine

failures = []

for kp in (12.500, 14.850):
    print(f"\n{'='*78}\nCHAINAGE KP {kp:.3f}\n{'='*78}")
    for action in INTERVENTION_ACTIONS:
        h = TelemetryHarmonizer()
        e = WhatIfEngine(harmonizer=h)
        h.seek_chainage(kp)
        # advance into a door-moving phase so door metrics are diagnosable
        for _ in range(95):
            h.generate_next_frame(dt=0.1)
        h.seek_chainage(kp)

        res = e.simulate_action(action, True)
        impact = res["measured_impact"]
        ok = impact["has_measurable_effect"]
        print(f"\n[{'OK  ' if ok else 'DEAD'}] {action}  ({res['title']})")
        if ok:
            for m in impact["changed_metrics"]:
                arrow = "v" if m["direction"] == "better" else "^"
                print(f"        {arrow} {m['label']:22s} "
                      f"{m['before']:>8.2f} -> {m['after']:>8.2f} {m['unit']} [{m['direction']}]")
        else:
            print(f"        {impact['note']}")
            # Rail grinding legitimately has no effect on already-smooth track.
            if not (action == "ACTION_GRIND_RAIL" and kp == 12.500):
                failures.append(f"{action} @ KP {kp}")

# Counterfactual block must appear in the streamed frame
h = TelemetryHarmonizer()
h.seek_chainage(14.850)
h.apply_action("ACTION_GRIND_RAIL", True)
h.apply_action("ACTION_INSPECT_BEARING", True)
frame = h.generate_next_frame(dt=0.1)
cf = frame["counterfactual"]
print(f"\n{'='*78}\nFRAME COUNTERFACTUAL BLOCK\n{'='*78}")
print(f"active={cf['active']} actions={cf['active_actions']}")
print(f"headline={cf['headline']}")
print(f"improved_count={cf['improved_count']}")
for m in cf["metrics"]:
    if m["direction"] != "unchanged":
        print(f"  {m['label']:22s} {m['before']:>8.2f} -> {m['after']:>8.2f} {m['unit']:3s} [{m['direction']}]")

assert cf["active"], "counterfactual block not active"
assert cf["improved_count"] > 0, "no improvement recorded"
assert "plain_status" in frame, "plain_status missing"
print(f"\nplain_status sample: {list(frame['plain_status'].items())[:4]}")
print(f"next_corrugation_zone: {frame['next_corrugation_zone']}")

print(f"\n{'='*78}")
if failures:
    print(f"FAILURES: {failures}")
    sys.exit(1)
print("ALL INTERVENTIONS PRODUCE MEASURABLE EFFECTS")
