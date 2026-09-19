"""
backend/models/submission.py
===============================================================================
The PS3 prediction-CSV each subsystem is scored on (see PS3/04_Example_Submission).
Models attach one of these to their result as `submission`, so the UI can offer
the exact file for download rather than re-deriving it from display fields.
===============================================================================
"""

from typing import Iterable, Optional


def make_submission(filename: str, columns: list, rows: list) -> dict:
    return {"filename": filename, "columns": columns, "rows": rows}


def merge_submissions(results: Iterable[dict]) -> Optional[dict]:
    """One submission for a batch upload: every file's rows under a shared header."""
    parts = [r["submission"] for r in results if r.get("submission")]
    if not parts:
        return None
    first = parts[0]
    return make_submission(first["filename"], first["columns"], [row for p in parts for row in p["rows"]])


def door_timestamp(ts) -> str:
    """Door dataset timestamp: Year-Month-Day-Hour-Minute-Second-Millisecond, not zero-padded."""
    return f"{ts.year}-{ts.month}-{ts.day}-{ts.hour}-{ts.minute}-{ts.second}-{ts.microsecond // 1000}"
