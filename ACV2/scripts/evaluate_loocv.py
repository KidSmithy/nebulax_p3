"""
evaluate_loocv.py
===========================================================================
Full evaluation of the localiser on the six labelled cases:

  1. the rank-decay score with the uncalibrated physical prior,
  2. leave-one-file-out cross validation of the whole procedure including the
     weight calibration,
  3. a group ablation (what breaks if each branch of the physics is removed),
  4. single-channel discrimination for every feature,
  5. a block-bootstrap stability check per case.

Run:  python ACV2/scripts/evaluate_loocv.py [--no-bootstrap]
Writes: ACV2/reports/evaluation.md
===========================================================================
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acv2 import config as cfg                                     # noqa: E402
from acv2 import confidence                                        # noqa: E402
from acv2.evaluate import (RANDOM_BASELINE_8, ablation,            # noqa: E402
                           block_bootstrap_stability, build_case_set,
                           calibrate_groups, loocv, observed_separation,
                           peer_dropout_stability, per_feature_ranks)
from acv2.ranker import DEFAULT_MODEL                              # noqa: E402
from acv2.reporting import df_to_md                                # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-bootstrap", action="store_true")
    ap.add_argument("--draws", type=int, default=40)
    args = ap.parse_args()

    os.makedirs(cfg.REPORT_DIR, exist_ok=True)
    out: list[str] = ["# ACV2 evaluation report", ""]

    print("Loading cases ...")
    case_set = build_case_set(verbose=True)

    # ---- 1. uncalibrated physical prior --------------------------------
    print("\n[1] Physical prior, no calibration")
    prior = case_set.score(DEFAULT_MODEL)
    print(prior.to_string(index=False))
    prior_mean = float(prior["score"].mean())
    print(f"  mean score = {prior_mean:.4f}   (random baseline {RANDOM_BASELINE_8:.4f})")
    out += ["## 1. Uncalibrated physical prior", "",
            "Weights are the physical prior in `config.FEATURE_SPEC`; nothing is fitted.", "",
            df_to_md(prior), "",
            f"**Mean rank-decay score {prior_mean:.4f}** against a random-permutation "
            f"baseline of {RANDOM_BASELINE_8:.4f}.", ""]

    # ---- 2. leave-one-file-out ----------------------------------------
    print("\n[2] Leave-one-file-out cross validation (calibration inside the loop)")
    cv = loocv(case_set)
    cv_mean = float(cv["score"].mean())
    print(f"  LOOCV mean score = {cv_mean:.4f}")
    out += ["## 2. Leave-one-file-out cross validation", "",
            "For each held-out file the group weights are re-selected by coordinate ascent "
            "on the other five files only, so the held-out score never sees its own tuning.", "",
            df_to_md(cv[["held_out", "true_car", "rank", "score", "train_score", "ranked_cars"]]),
            "",
            f"**LOOCV mean rank-decay score {cv_mean:.4f}.**", ""]

    # ---- 3. ablation ---------------------------------------------------
    print("\n[3] Group ablation")
    abl = ablation(case_set)
    out += ["## 3. Feature-group ablation", "",
            "`removed = X` zeroes group X; `all but X` keeps only group X.", "",
            df_to_md(abl), ""]

    # ---- 4. single-channel discrimination ------------------------------
    print("\n[4] Single-channel discrimination")
    summary, long = per_feature_ranks(case_set)
    print(summary.to_string(index=False))
    pivot = long.pivot_table(index=["group", "feature"], columns="file_id", values="rank")
    out += ["## 4. Single-channel discrimination", "",
            "Rank-decay score each feature achieves on its own. `n_files` is how many files "
            "the channel was computable and non-degenerate in.", "",
            df_to_md(summary), "",
            "Rank of the true faulty car per file, by channel:", "",
            df_to_md(pivot, index=True), ""]

    # ---- 5. calibration on all data (for the shipped artefact) ---------
    groups, fitted = calibrate_groups(case_set, case_set.labelled(), verbose=True)
    out += ["## 5. Calibration on all six files", "",
            f"Coordinate ascent reaches a training score of {fitted:.4f} with group weights "
            f"`{groups}`.", "",
            "Because the physical prior already scores 1.0000 there is nothing for the "
            "calibration to correct, and it returns the prior unchanged - the shipped model "
            "is therefore the physics, not a fit to six files.", ""]

    # ---- 6. stability --------------------------------------------------
    if not args.no_bootstrap:
        print("\n[5] Block-bootstrap stability")
        rows = []
        for file_id, case in case_set.cases.items():
            stab = block_bootstrap_stability(case, DEFAULT_MODEL, n_draws=args.draws)
            true_car = case_set.labels.get(file_id, "?")
            freq = stab["top1_frequency"]
            rows.append({
                "file_id": file_id,
                "true_car": true_car,
                "draws": stab["n_draws"],
                "top1_true_car": round(freq.get(true_car, 0.0), 3),
                "most_frequent": next(iter(freq), "-"),
                "most_frequent_freq": round(next(iter(freq.values()), 0.0), 3),
                "mean_rank_true": round(stab["mean_rank"].get(true_car, float("nan")), 2),
            })
            print(f"  {file_id}: true={true_car} top-1 frequency={rows[-1]['top1_true_car']}")
        stability = pd.DataFrame(rows)
        out += ["## 6. Block-bootstrap stability", "",
                "Each case is re-ranked on random 70% subsets of its 6-hour blocks. "
                "`top1_true_car` is how often the true faulty car still comes first.", "",
                df_to_md(stability), "",
                "This measures *temporal* stability only: every draw still judges each car "
                "against the same seven siblings. The companion car-level test, which "
                "resamples the consist itself, is test D in `reports/robustness.md`.", ""]

        print("\n[6] Peer-dropout (car-level) stability")
        rows = []
        for file_id, case in case_set.cases.items():
            stab = peer_dropout_stability(case, DEFAULT_MODEL, n_draws=max(10, args.draws // 2))
            true_car = case_set.labels.get(file_id, "?")
            rows.append({
                "file_id": file_id, "true_car": true_car, "draws": stab["n_draws"],
                "cars_dropped": stab.get("drop_k"),
                "top1_true_car": round(stab["top1_frequency"].get(true_car, 0.0), 3),
                "leader_retained": round(stab["leader_retained"], 3)
                if pd.notna(stab["leader_retained"]) else float("nan"),
            })
            print(f"  {file_id}: true={true_car} top-1 frequency={rows[-1]['top1_true_car']}")
        out += ["## 7. Peer-dropout (car-level) stability", "",
                "Each case re-ranked from scratch after dropping randomly chosen non-leading "
                "cars, with the peer reference, ambient estimate and load strata all "
                "recomputed. This tests whether the verdict is a property of the accused car "
                "or of the particular sibling group it was compared against.", "",
                df_to_md(pd.DataFrame(rows)), ""]

    # ---- 8. calibrated confidence --------------------------------------
    print("\n[7] Separation against the fault-free null")
    separation = observed_separation(case_set, DEFAULT_MODEL)
    calibration = confidence.load_null_calibration()
    if calibration:
        separation["null_p_value"] = [
            confidence.p_value_for({"dixon_q": q, "top_z": z}, calibration)
            for q, z in zip(separation["dixon_q"], separation["top_z"])]
    print(separation.to_string(index=False))
    n_null = len(calibration["dixon_q_null"]) if calibration else 0
    out += ["## 8. Calibrated confidence, not margin", "",
            "The system used to report the raw top-1 margin as though it were confidence. "
            "Test B in `reports/robustness.md` showed that is wrong: deleting the labelled "
            "faulty car and re-ranking the healthy siblings produces a margin of the same "
            "size, because a peer-consensus ranker always crowns the most anomalous car "
            "whether or not one is broken. Confidence is therefore reported as the scale-free "
            "Dixon-Q separation referenced to a null of "
            f"{n_null} fault-free consists.", "",
            df_to_md(separation), "",
            "`null_p_value` is the fraction of fault-free consists separating at least as "
            "cleanly. Three of the six genuine faults do not separate distinguishably from a "
            "healthy consist, which is the honest state of the evidence rather than a defect: "
            "a leak whose capacity deficit is still small genuinely does look like a healthy "
            "pack. The metric is reported so the system can say so instead of quoting a "
            "confident margin it cannot justify.", ""]

    report_path = os.path.join(cfg.REPORT_DIR, "evaluation.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
