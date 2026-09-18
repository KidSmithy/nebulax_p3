"""
backend/services/demo_seed.py
===============================================================================
Seeds the NSL line with one finding per subsystem at startup, produced by
running the real models on files from the PS3 training/test data - exactly as if
someone had uploaded them through the Conductor. That way the 3D view opens with
all four issue bubbles and the cards are grounded in the project's own data.

Disable with NSL_DEMO_FINDINGS=0. A file that fails to score is skipped, never
fatal. EWL is left empty on purpose so it can be demoed with a live upload.
===============================================================================
"""

import os

from backend.core.config import DATASETS_DIR

NSL_DEMO_ENABLED = os.getenv("NSL_DEMO_FINDINGS", "1").strip().lower() not in ("0", "false", "no")

# subsystem -> file under PS3/02_Datasets. All held-out Test files (none of the models saw them
# in training), chosen so the four bubbles show a spread of severities.
NSL_DEMO_FILES = {
    "acv": "ACV/Test/acv_test_case.xlsx",
    "door": "Door/Test.csv",
    "shm": "SHM/Test/test01.csv",
    "rail": "Rail_Corrugation/Test/Test2.csv",
}


def seed_nsl_findings(harmonizer) -> None:
    models = {
        "acv": harmonizer.broker.acv_model,
        "door": harmonizer.broker.door_model,
        "shm": harmonizer.broker.shm_model,
        "rail": harmonizer.broker.rail_model,
    }
    for subsystem, rel_path in NSL_DEMO_FILES.items():
        path = DATASETS_DIR / rel_path
        try:
            result = models[subsystem].predict_from_csv(path.read_bytes(), path.name)
            harmonizer.set_upload_result(subsystem, result, line="NSL")
            print(f"[DemoSeed] NSL {subsystem}: {result.get('status')} from {path.name}")
        except Exception as exc:
            print(f"[DemoSeed] NSL {subsystem} skipped ({path.name}): {type(exc).__name__}: {exc}")
