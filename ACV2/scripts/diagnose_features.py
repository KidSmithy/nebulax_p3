"""
diagnose_features.py
===========================================================================
Show what each physical channel actually contributes.

For every training case it prints the per-car feature table, the oriented
z-scores, where the true faulty car ranks under each feature used alone, and
the fused ranking. This is the script to run when a change to the physics needs
justifying.

Run:  python ACV2/scripts/diagnose_features.py
===========================================================================
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acv2 import config as cfg                                  # noqa: E402
from acv2.detectors import fuse, oriented_z                     # noqa: E402
from acv2.evaluate import (RANDOM_BASELINE_8, build_case_set,   # noqa: E402
                           per_feature_ranks, rank_decay_score, rank_of)
from acv2.ranker import DEFAULT_MODEL                           # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)
pd.set_option("display.float_format", lambda v: f"{v:8.3f}")


def main() -> None:
    case_set = build_case_set(verbose=True)

    print("\n" + "=" * 110)
    print("PER-CASE FEATURE TABLES AND FUSED RANKING")
    print("=" * 110)
    for file_id in case_set.labelled():
        features = case_set.features[file_id]
        true_car = case_set.labels[file_id]
        result = fuse(features,
                      group_weights=DEFAULT_MODEL["group_weights"],
                      feature_weights=DEFAULT_MODEL["feature_weights"])
        ranked = case_set.ranked(file_id, DEFAULT_MODEL)
        print("\n" + "-" * 110)
        print(f"{file_id}  true_faulty={true_car}  schema={case_set.cases[file_id].report['schema']}  "
              f"rank={rank_of(ranked, true_car)}  score={rank_decay_score(ranked, true_car):.3f}")
        print(f"  ranked: {'|'.join(ranked)}")

        shown = [c for c in features.columns
                 if c in result["z"].columns]
        display = features[shown].copy()
        display["FAULTY"] = ["<<<" if c == true_car else "" for c in display.index]
        print("\n  raw features:")
        print(display.to_string())
        z = result["z"].copy()
        z["SCORE"] = result["score"]
        z["FAULTY"] = ["<<<" if c == true_car else "" for c in z.index]
        print("\n  oriented z-scores (positive = leak-like):")
        print(z.to_string())

    print("\n" + "=" * 110)
    print("SINGLE-FEATURE DISCRIMINATION (rank-decay score of each channel used alone)")
    print("=" * 110)
    summary, long = per_feature_ranks(case_set)
    print(summary.to_string(index=False))
    print(f"\n  random-permutation baseline for an 8-car file = {RANDOM_BASELINE_8:.4f}")

    print("\n  per-file rank of the true faulty car, by feature:")
    pivot = long.pivot_table(index=["group", "feature"], columns="file_id", values="rank")
    print(pivot.to_string())

    print("\n" + "=" * 110)
    print("FUSED RESULT WITH THE PHYSICAL PRIOR (no calibration)")
    print("=" * 110)
    scored = case_set.score(DEFAULT_MODEL)
    print(scored.to_string(index=False))
    print(f"\n  mean rank-decay score = {scored['score'].mean():.4f}  "
          f"(random baseline {RANDOM_BASELINE_8:.4f})")


if __name__ == "__main__":
    main()
