"""Live test of the AI insight service against the configured model."""
import sys, pathlib, json, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from backend.core.harmonizer import TelemetryHarmonizer
from backend.services.ai_insight import AIInsightService, build_frame_summary

h = TelemetryHarmonizer()
h.seek_chainage(14.850)
for _ in range(95):
    h.generate_next_frame(dt=0.1)
    h.generator.current_kp = 14.850
frame = h.generate_next_frame(dt=0.1)

svc = AIInsightService()
print("STATUS:", json.dumps(svc.status(), indent=2))
print("\n--- BRIEFING SENT TO MODEL ---")
print(build_frame_summary(frame))

t0 = time.time()
ins = svc.get_insight(frame)
print(f"\n--- INSIGHT ({time.time()-t0:.1f}s) ---")
print(json.dumps(ins, indent=2, ensure_ascii=False))

t0 = time.time()
ans = svc.ask("Is it safe for passengers right now, and what should I fix first?", frame)
print(f"\n--- ASK ({time.time()-t0:.1f}s) ---")
print(json.dumps(ans, indent=2, ensure_ascii=False))

assert ins.get("source") == "openai", f"AI call did not succeed: {ins.get('degraded_reason')}"
assert ins.get("severity") in ("GOOD", "WATCH", "ACTION_NEEDED")
assert ans.get("source") == "openai", f"Ask did not succeed: {ans.get('degraded_reason')}"
print("\nAI SERVICE OK")
