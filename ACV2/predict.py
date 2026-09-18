"""
predict.py
===========================================================================
Inference entry point for the ACV refrigerant-leak localisation task.

    python ACV2/predict.py --input <file-or-directory> --output acv_predictions.csv

Produces one row per input case file:

    file_id,ranked_cars
    acv_test_case.xlsx,04|01|03|...

``ranked_cars`` lists *every* car that appears in that file's own headers, most
to least likely faulty, using the identifier exactly as the headers spell it
(``03``, not ``Car 3``), pipe separated.

Optional:
    --report     also write a human-readable diagnosis per file
    --model      path to a weight artefact (defaults to artifacts/acv2_model.joblib,
                 and falls back to the physical prior if there is none)
===========================================================================
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from acv2 import config as cfg                      # noqa: E402
from acv2.io_loader import discover_cases           # noqa: E402
from acv2.ranker import load_model, rank_case       # noqa: E402

warnings.filterwarnings("ignore")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Localise refrigerant leakage to a car.")
    ap.add_argument("--input", "-i", required=True,
                    help="case file (.xlsx/.csv) or a directory of case files")
    ap.add_argument("--output", "-o", default="acv_predictions.csv",
                    help="output CSV path (default: acv_predictions.csv)")
    ap.add_argument("--model", "-m", default=None,
                    help="weight artefact; defaults to ACV2/artifacts/acv2_model.joblib")
    ap.add_argument("--report", action="store_true",
                    help="print a per-car diagnosis and write a JSON sidecar")
    ap.add_argument("--quiet", "-q", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not os.path.exists(args.input):
        print(f"[predict] input not found: {args.input}", file=sys.stderr)
        return 2
    files = discover_cases(args.input)
    if not files:
        print(f"[predict] no .xlsx/.csv case files under {args.input}", file=sys.stderr)
        return 2

    model = load_model(args.model)
    if not args.quiet:
        source = args.model or cfg.MODEL_PATH
        state = "calibrated artefact" if model.get("calibrated") else "physical prior"
        print(f"[predict] model: {state} ({source if os.path.exists(source) else 'built-in'})")

    rows, details = [], []
    for path in files:
        result = rank_case(path, model=model, verbose=not args.quiet)
        rows.append({"file_id": result["file_id"], "ranked_cars": result["ranked_cars"]})
        details.append(result)
        if not args.quiet:
            print(f"[predict] {result['file_id']}: {result['ranked_cars']}")
        if args.report:
            print(f"\n  {result['verdict']}")
            print(f"  schema={result['schema']}  evidence groups={', '.join(result['active_groups'])}")
            print(f"  {'rank':>4}  {'car':>3}  {'score':>7}  {'physics':>7}  evidence")
            for entry in result["car_table"]:
                evidence = ", ".join(f"{e['feature']}={e['value']:.3g}" for e in entry["evidence"][:3])
                print(f"  {entry['rank']:>4}  {entry['car']:>3}  {entry['score']:>7.3f}  "
                      f"{entry['physics_score']:>7.3f}  {evidence or '-'}")
            print()

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file_id", "ranked_cars"])
        writer.writeheader()
        writer.writerows(rows)
    if not args.quiet:
        print(f"[predict] wrote {len(rows)} row(s) to {os.path.abspath(args.output)}")

    if args.report:
        sidecar = os.path.splitext(args.output)[0] + "_diagnosis.json"
        payload = [{
            "file_id": d["file_id"],
            "ranked_cars": d["ranked_cars"],
            "most_likely_faulty_car": d["most_likely_faulty_car"],
            "schema": d["schema"],
            "active_groups": d["active_groups"],
            "verdict": d["verdict"],
            "cars": [{k: v for k, v in entry.items()} for entry in d["car_table"]],
        } for d in details]
        with open(sidecar, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        if not args.quiet:
            print(f"[predict] wrote {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
