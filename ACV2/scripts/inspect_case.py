"""
inspect_case.py
===========================================================================
Deep inspection of a single case file: cleaning report, per-car physics
features, oriented z-scores, weighted contributions, the daily elevation
profile and a block-bootstrap confidence on the verdict.

Run:  python ACV2/scripts/inspect_case.py PS3/02_Datasets/ACV/Test/acv_test_case.xlsx
===========================================================================
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acv2 import config as cfg                                  # noqa: E402
from acv2.cleaning import clean                                 # noqa: E402
from acv2.evaluate import block_bootstrap_stability             # noqa: E402
from acv2.features import build_context, extract                # noqa: E402
from acv2.io_loader import load_case, load_labels               # noqa: E402
from acv2.ranker import load_model, rank_case                   # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 80)
pd.set_option("display.float_format", lambda v: f"{v:9.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--draws", type=int, default=60)
    args = ap.parse_args()

    panel = load_case(args.path, verbose=True)
    case = clean(panel)
    print("\n" + case.summary())
    true_car = load_labels().get(panel.file_id)
    if true_car:
        print(f"ground truth faulty car: {true_car}")

    features, ctx = extract(case)
    model = load_model()
    result = rank_case(case, model=model)

    def mark(index):
        return ["<<<" if c == true_car else "" for c in index]

    print("\n--- per-car physics features " + "-" * 70)
    shown = features[[c for c in features.columns if not c.startswith("_")]].dropna(axis=1, how="all")
    shown = shown.assign(FAULTY=mark(shown.index))
    print(shown.to_string())

    print("\n--- diagnostics " + "-" * 82)
    diag = features[[c for c in features.columns if c.startswith("_")]]
    print(diag.to_string())

    print("\n--- oriented z-scores (positive = leak-like) " + "-" * 54)
    z = result["z_scores"].assign(FAULTY=mark(result["z_scores"].index))
    print(z.to_string())

    print("\n--- weighted contributions to the score " + "-" * 59)
    contrib = result["contributions"].copy()
    contrib["TOTAL"] = contrib.sum(axis=1)
    contrib["FAULTY"] = mark(contrib.index)
    print(contrib.sort_values("TOTAL", ascending=False).to_string())
    print("\n  weights in use: " + ", ".join(f"{k}={v:.2f}" for k, v in result["weights"].items()))

    print("\n--- ranking " + "-" * 86)
    for entry in result["car_table"]:
        flag = "  <<< TRUE FAULTY" if entry["car"] == true_car else ""
        print(f"  {entry['rank']:>2}. car {entry['car']}  score={entry['score']:+.3f}  "
              f"physics={entry['physics_score']:+.3f}  outlier={entry['outlier_score']:+.3f}{flag}")
    print("\n  " + result["verdict"])

    print("\n--- daily mean peer-relative elevation (K) " + "-" * 55)
    elev = ctx["elevation"]
    day = case.time.dt.floor("D")
    profile = elev.groupby(day).mean()
    profile.index = [str(d.date()) for d in profile.index]
    print(profile.T.to_string())

    print("\n--- block-bootstrap stability " + "-" * 68)
    stab = block_bootstrap_stability(case, model, n_draws=args.draws)
    print(f"  draws={stab['n_draws']}")
    print("  top-1 frequency: " + ", ".join(f"{c}={f:.2f}" for c, f in stab["top1_frequency"].items()))
    print("  mean rank:       " + ", ".join(f"{c}={r:.2f}" for c, r in stab["mean_rank"].items()))

    print("\n--- sensitivity: ranking with each evidence group removed " + "-" * 40)
    base_groups = dict(model.get("group_weights", cfg.GROUP_WEIGHTS))
    for removed in ("(nothing)",) + cfg.FEATURE_GROUPS:
        groups = dict(base_groups)
        if removed in groups:
            groups[removed] = 0.0
        trial = dict(model, group_weights=groups)
        ranking = rank_case(case, model=trial)["ranked_cars"]
        print(f"  without {removed:<12} -> {ranking}")


if __name__ == "__main__":
    main()
