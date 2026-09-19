"""
backend/scripts/build_seed_findings.py
===============================================================================
Runs the four real subsystem models over real PS3 dataset files ONCE and saves
their unedited output envelopes to backend/model_data/seed_findings.json.

The backend replays that file at startup so a fresh page load already shows all
four inspection bubbles. See backend/core/seed_findings.py for why this is done
offline instead of at boot.

Usage (from the project root):

    .\\venv\\Scripts\\python.exe -m backend.scripts.build_seed_findings

    # only refresh some subsystems
    .\\venv\\Scripts\\python.exe -m backend.scripts.build_seed_findings door shm
===============================================================================
"""

import datetime as dt
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from backend.core.seed_findings import SEED_FILE, SEED_ORDER, SEED_SOURCES
from backend.models.inference_broker import InferenceBroker


def _jsonable(obj):
    """numpy scalars/arrays are not JSON-serialisable; model output contains both."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON-serialisable: {type(obj)}")


def main(argv):
    wanted = [a.lower() for a in argv] or SEED_ORDER
    unknown = [w for w in wanted if w not in SEED_SOURCES]
    if unknown:
        print(f"Unknown subsystem(s): {', '.join(unknown)}. "
              f"Choose from: {', '.join(SEED_ORDER)}")
        return 2

    broker = InferenceBroker()
    models = {
        "door": broker.door_model,
        "acv": broker.acv_model,
        "shm": broker.shm_model,
        "rail": broker.rail_model,
    }

    # Keep any subsystem we are not regenerating this run.
    existing = {}
    if SEED_FILE.exists():
        try:
            existing = json.loads(SEED_FILE.read_text(encoding="utf-8")).get("findings", {})
        except (OSError, json.JSONDecodeError):
            existing = {}

    findings = dict(existing)
    sources = {}
    failures = []

    for sub in SEED_ORDER:
        if sub not in wanted:
            continue
        rel = SEED_SOURCES[sub]
        path = ROOT_DIR / rel
        if not path.exists():
            print(f"[{sub}] SKIP - dataset not found: {rel}")
            failures.append(sub)
            continue

        print(f"[{sub}] running model on {rel} ({path.stat().st_size / 1e6:.1f} MB) ...")
        try:
            result = models[sub].predict_from_csv(path.read_bytes(), path.name)
        except Exception as err:  # a bad seed must not be silently half-written
            print(f"[{sub}] FAILED - {type(err).__name__}: {err}")
            failures.append(sub)
            continue

        findings[sub] = result
        sources[sub] = rel
        print(f"[{sub}] -> status={result.get('status')} "
              f"score={result.get('anomaly_score')} "
              f"action={result.get('recommended_action')}")

    if not findings:
        print("Nothing to write.")
        return 1

    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "note": (
            "Real, unedited output of the four subsystem models. Regenerate with "
            "`python -m backend.scripts.build_seed_findings`."
        ),
        "sources": {**{k: v for k, v in SEED_SOURCES.items() if k in findings}, **sources},
        "findings": findings,
    }

    SEED_FILE.parent.mkdir(parents=True, exist_ok=True)
    SEED_FILE.write_text(
        json.dumps(payload, indent=2, default=_jsonable, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nWrote {SEED_FILE.relative_to(ROOT_DIR)} "
          f"({SEED_FILE.stat().st_size / 1024:.1f} KB) "
          f"with: {', '.join(findings)}")
    if failures:
        print(f"Incomplete - failed/skipped: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
