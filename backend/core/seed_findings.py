"""
backend/core/seed_findings.py
===============================================================================
Pre-seeded demo findings, so a fresh page load already shows all four
inspection bubbles instead of an empty train.

WHY A SAVED FILE RATHER THAN LIVE INFERENCE AT STARTUP
------------------------------------------------------
The four subsystem models are perfectly capable of running at boot, but their
real inputs are large (a single Rail_Corrugation CSV is ~17 MB with 128+
channels; an SHM run is ~6 MB) and the PS3/ dataset tree is not shipped with
the deployed backend. Doing it live would add tens of seconds to every cold
start and would break in any environment without the datasets.

So inference happens ONCE, offline:

    python -m backend.scripts.build_seed_findings

That script feeds real files from PS3/02_Datasets through the real models and
writes their real, unedited output envelopes to
`backend/model_data/seed_findings.json`. This module just replays that file at
startup. The numbers in the UI are therefore genuine model predictions - the
only thing "hardcoded" is which dataset file was analysed, exactly as if an
operator had uploaded it a moment before you opened the page.

Delete or disable the file (SEED_DEMO_FINDINGS=0) and the app returns to its
original empty-until-you-upload behaviour.
===============================================================================
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
SEED_FILE = ROOT_DIR / "backend" / "model_data" / "seed_findings.json"

# Which real dataset file stands in for each subsystem's "already uploaded"
# inspection run. Chosen for a demo that actually shows something:
#   door  - the Door TRAIN split, deliberately not the test split: Test.csv is
#           kept clean for the live upload walkthrough. Note this is the RAW
#           Train.csv, not Train_cleaned.csv - see the warning below.
#   acv   - labelled leak case, faulty car 04 (matches the UI's DEFAULT_CAR, so
#           the camera does not jump on first load)
#   shm   - train31, the highest labelled damage in Train_Labels.csv (0.928)
#   rail  - Train2, labelled "Side II" corrugation rather than "Normal"
#
# WARNING - do not "upgrade" the door entry to Door/*_cleaned.csv.
# The `_cleaned` variants were normalised for a different (notebook) pipeline
# and use snake_case headers (`motor_current_ma`, `datetime_str`, ...).
# door_model.predict_from_csv has its own column normaliser, but its alias
# table targets the ORIGINAL LTA headers (`Motor current(mA)`, `Datetime`, ...),
# and none of the cleaned names match it. Feeding it a cleaned file resolves
# 0/17 channels: motor current survives only by the "first numeric column"
# fallback, the other 16 channels are replaced by hardcoded constants, and with
# no `Datetime` match the cycle segmenter degrades from real timestamp-gap
# detection to blind 150-row chunks (121 fabricated cycles instead of 110 real
# ones). The raw Train.csv resolves 17/17 and reproduces
# Train_Segments_Answer.csv exactly: 110 cycles, 30 abnormal.
SEED_SOURCES: Dict[str, str] = {
    "door": "PS3/02_Datasets/Door/Train.csv",
    "acv": "PS3/02_Datasets/ACV/Train/acv_case_05.xlsx",
    "shm": "PS3/02_Datasets/SHM/Train/train31.csv",
    "rail": "PS3/02_Datasets/Rail_Corrugation/Train/Train2.csv",
}

# Order matters: ACV is seeded first because its finding names the faulty car
# that every other bubble is hung on (see useMonitoredItems/TrainAssembly).
SEED_ORDER: List[str] = ["acv", "door", "shm", "rail"]


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def seeding_enabled() -> bool:
    """Seeding is on by default; set SEED_DEMO_FINDINGS=0 for a blank start."""
    return _env_flag("SEED_DEMO_FINDINGS", True)


def seeded_lines() -> List[str]:
    """Lines that get the pre-seeded findings. Default: NSL only."""
    raw = os.getenv("SEED_DEMO_LINES", "NSL")
    return [part.strip().upper() for part in raw.split(",") if part.strip()]


def load_seed_findings() -> Optional[Dict[str, dict]]:
    """
    The saved model output, as `{subsystem: finding}` in SEED_ORDER.

    Returns None (and never raises) if the file is missing or unreadable - a
    missing seed must degrade to today's empty-state behaviour, not a crash on
    startup.
    """
    if not SEED_FILE.exists():
        return None
    try:
        payload = json.loads(SEED_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        print(f"[Seed] Ignoring unreadable {SEED_FILE.name}: {err}")
        return None

    findings = payload.get("findings")
    if not isinstance(findings, dict):
        print(f"[Seed] Ignoring {SEED_FILE.name}: no 'findings' object.")
        return None

    ordered = {sub: findings[sub] for sub in SEED_ORDER if isinstance(findings.get(sub), dict)}
    return ordered or None
