"""
predict.py -- ACV refrigerant-leakage localisation (inference entry point)
==============================================================================
Official CLI:

    python predict.py --input <case_file_or_directory> --output <predictions.csv>

Examples:
    python predict.py --input ../PS3/02_Datasets/ACV/Test --output acv_predictions.csv
    python predict.py --input ../PS3/02_Datasets/ACV/Test/acv_test_case.xlsx \
                      --output acv_predictions.csv

Output schema (one row per input case file):
    file_id      source file name including extension, e.g. acv_test_case.xlsx
    ranked_cars  every car identifier in that file's own headers, ordered most
                 to least likely faulty, pipe separated, e.g. 01|03|04|...

There is no model artefact to load. The ranker is a closed-form physics index
(see acv_physics_ranker.py), so inference is deterministic and depends only on
the file being scored -- no training data, no pickle, no version drift.
==============================================================================
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from acv_physics_ranker import rank_consist  # noqa: E402

CASE_PATTERNS = ("*.xlsx", "*.xls", "*.csv")


def collect_inputs(target: str) -> list[str]:
    if os.path.isdir(target):
        files: list[str] = []
        for pattern in CASE_PATTERNS:
            files.extend(glob.glob(os.path.join(target, pattern)))
        # Skip Excel lock/temp files.
        return sorted(f for f in files if not os.path.basename(f).startswith("~$"))
    if os.path.isfile(target):
        return [target]
    raise FileNotFoundError(f"--input path does not exist: {target}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Localise the refrigerant-leaking car in ACV consist telemetry."
    )
    ap.add_argument("--input", required=True,
                    help="Case .xlsx/.csv file, or a directory of case files.")
    ap.add_argument("--output", required=True,
                    help="Path of the acv_predictions.csv to write.")
    ap.add_argument("--verbose", action="store_true",
                    help="Print the per-car diagnostic table for each case.")
    args = ap.parse_args(argv)

    paths = collect_inputs(args.input)
    if not paths:
        print(f"[predict] no case files found in {args.input}", file=sys.stderr)
        return 1

    records = []
    for path in paths:
        try:
            res = rank_consist(path)
        except Exception as exc:                        # noqa: BLE001
            print(f"[predict] FAILED {os.path.basename(path)}: {exc}", file=sys.stderr)
            continue

        records.append({"file_id": res["file_id"], "ranked_cars": res["ranked_cars"]})
        print(f"[predict] {res['file_id']}: {res['ranked_cars']}"
              f"   (leak -> Car {res['most_likely_faulty_car']},"
              f" margin {res['confidence_margin']})")

        if args.verbose:
            print(pd.DataFrame(res["car_diagnostics"]).to_string(index=False))
            print(res["diagnostic_summary"])
            print()

    if not records:
        print("[predict] no case produced a prediction", file=sys.stderr)
        return 1

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(records, columns=["file_id", "ranked_cars"]).to_csv(
        args.output, index=False
    )
    print(f"[predict] wrote {len(records)} row(s) to {os.path.abspath(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
