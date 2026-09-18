"""Check fleet health calibration: baseline vs fully maintained."""
import sys, pathlib, statistics
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from backend.core.config import INTERVENTION_ACTIONS, classify_metric
from backend.core.harmonizer import TelemetryHarmonizer


def sample(kp, actions, n=260):
    h = TelemetryHarmonizer()
    h.seek_chainage(kp)
    for a in actions:
        h.apply_action(a, True)
    vals, door, shm, acv = [], [], [], []
    for _ in range(n):
        f = h.generate_next_frame(dt=0.1)
        h.generator.current_kp = kp  # hold position
        vals.append(f["fleet_health_index"])
        s = f["subsystems"]
        door.append(s["door"]["anomaly_score"])
        shm.append(s["shm"]["vibration_rms_g"])
        acv.append(s["acv"]["efficiency_rating"])
    return vals, door, shm, acv


for kp in (12.500, 14.850):
    print(f"\n=== KP {kp:.3f} ===")
    for label, actions in [
        ("baseline (no maintenance)", []),
        ("all maintenance applied", INTERVENTION_ACTIONS),
    ]:
        v, d, s, a = sample(kp, actions)
        print(f"{label:28s} health min={min(v):.2f} mean={statistics.mean(v):.2f} max={max(v):.2f} "
              f"[{classify_metric('fleet_health_index', statistics.mean(v))}]")
        print(f"{'':28s} door_anom max={max(d):.2f} | vib mean={statistics.mean(s):.2f}g "
              f"| acv_eff={statistics.mean(a):.2f}")
