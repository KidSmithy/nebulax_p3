"""
backend/services/acv_service.py
===============================================================================
Runs an uploaded ACV case workbook through the ACV2 refrigerant-leak ranker.
===============================================================================
"""

import math
import os
import sys
import tempfile
from pathlib import Path

ACV2_DIR = Path(__file__).resolve().parents[2] / "ACV2"
if str(ACV2_DIR) not in sys.path:
    sys.path.insert(0, str(ACV2_DIR))

from acv2.ranker import load_model, rank_case  # noqa: E402

ALLOWED_SUFFIXES = {".xlsx", ".csv"}
MAX_UPLOAD_BYTES = 100 * 1024 * 1024


def _plain(value):
    """numpy scalars / NaN -> JSON-safe python values."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class UploadError(ValueError):
    """The upload itself is unacceptable (type/size), as opposed to unreadable contents."""


class ACVService:
    def __init__(self):
        self.model = load_model()

    def predict(self, filename: str, data: bytes) -> dict:
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise UploadError("Upload an ACV case file (.xlsx or .csv).")
        if len(data) > MAX_UPLOAD_BYTES:
            raise UploadError("That file is too large (100 MB max).")

        # Written under its own name so the result's file_id matches what was uploaded.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, Path(filename).name)
            with open(path, "wb") as fh:
                fh.write(data)
            result = rank_case(path, model=self.model)

        return {
            "file_id": result["file_id"],
            "ranked_cars": result["ranked_cars"],
            "ranked_cars_list": [str(c) for c in result["ranked_cars_list"]],
            "most_likely_faulty_car": str(result["most_likely_faulty_car"]),
            "verdict": result["verdict"],
            "cars": [
                {
                    "rank": int(entry["rank"]),
                    "car": str(entry["car"]),
                    "score": _plain(entry["score"]),
                    "diagnosable": bool(entry["diagnosable"]),
                    "evidence": [
                        {"feature": e["feature"], "value": _plain(e["value"])}
                        for e in entry["evidence"][:3]
                    ],
                }
                for entry in result["car_table"]
            ],
        }
