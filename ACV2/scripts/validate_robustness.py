"""
validate_robustness.py
===========================================================================
Three tests that the standard evaluation cannot perform, each aimed at a
specific, known weakness of a six-file problem.

A. WEIGHT SENSITIVITY  -- "the physical prior was hand-designed after looking
   at all six files, so LOOCV overstates generalisation."
   True. So: is the result a property of *those particular weights*, or of the
   physics? Score the six files under hundreds of random non-negative weight
   vectors over the same channels, keeping only the physically-fixed signs. If
   almost any positive weighting works, the hand-chosen numbers are not what is
   doing the work and the hand-tuning objection loses most of its force.

B. SPECIFICITY / NULL TEST -- "the bootstrap measures within-case stability,
   not whether a confident verdict is meaningful."
   So: delete the known faulty car and re-run on the seven healthy siblings.
   A detector whose confidence means anything must lose it here. If the top-1
   margin among seven healthy cars is as large as it was with the faulty car
   present, the score is just crowning the warmest car and the confidence is
   theatre.

C. SYNTHETIC RECOVERY AND DETECTION LIMIT -- "how far can we trust a verdict
   built on a +0.11 K elevation?"
   So: remove the real faulty car, inject a physics-shaped capacity deficit of
   known magnitude into one of the remaining healthy cars, and measure how often
   it is recovered as a function of that magnitude. Because the injected car is
   known and the real fault is gone, every trial is a genuinely unseen case.

   Honest limit of test C: the injection follows the same load-scaled,
   progressive shape the features are built around, so it measures *sensitivity
   to a deficit of a given size in real telemetry noise* - not whether the leak
   model itself is right. It is quantised onto the channel's own measurement grid
   so a sub-resolution deficit is not made artificially easy.

Run:  python ACV2/scripts/validate_robustness.py [--draws 300] [--trials 3]
Writes: ACV2/reports/robustness.md
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

from acv2 import config as cfg                                      # noqa: E402
from acv2.detectors import fuse, informative_features               # noqa: E402
from acv2.evaluate import (RANDOM_BASELINE_8, build_case_set,       # noqa: E402
                           rank_decay_score)
from acv2.features import extract                                   # noqa: E402
from acv2.ranker import DEFAULT_MODEL                               # noqa: E402
from acv2.reporting import df_to_md                                 # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)


def _rank(features: pd.DataFrame, header_cars: list[str], model: dict) -> tuple[list[str], float]:
    result = fuse(features,
                  group_weights=model.get("group_weights"),
                  feature_weights=model.get("feature_weights"),
                  w_physics=model.get("w_physics"),
                  w_outlier=model.get("w_outlier"))
    score = result["score"].sort_values(ascending=False)
    order = score.index.tolist()
    margin = float(score.iloc[0] - score.iloc[1]) if len(score) > 1 else float("nan")
    return order + [c for c in header_cars if c not in order], margin


# --------------------------------------------------------------------------
# A. weight sensitivity
# --------------------------------------------------------------------------
def weight_sensitivity(case_set, n_draws: int, seed: int = 20260919) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    channels = sorted({name for fid in case_set.labelled()
                       for name in informative_features(case_set.features[fid])})
    print(f"  sampling {n_draws} random weight vectors over {len(channels)} channels")

    rows = []
    for draw in range(n_draws):
        # Uniform on the simplex: no channel is ever negative (the signs are
        # physics), but the relative emphasis is arbitrary.
        weights = dict(zip(channels, rng.dirichlet(np.ones(len(channels)))))
        model = dict(DEFAULT_MODEL, feature_weights=weights,
                     group_weights={g: 1.0 for g in cfg.FEATURE_GROUPS})
        scores = []
        for file_id in case_set.labelled():
            ranked, _margin = _rank(case_set.features[file_id],
                                    case_set.header_cars(file_id), model)
            scores.append(rank_decay_score(ranked, case_set.labels[file_id]))
        rows.append({"draw": draw, "mean_score": float(np.mean(scores)),
                     "n_top1": int(sum(1 for s in scores if s == 1.0))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# B. specificity: no faulty car present
# --------------------------------------------------------------------------
def specificity(case_set, model: dict) -> pd.DataFrame:
    rows = []
    for file_id in case_set.labelled():
        case = case_set.cases[file_id]
        true_car = case_set.labels[file_id]

        ranked, margin_with = _rank(case_set.features[file_id],
                                    case_set.header_cars(file_id), model)
        healthy = case.without_cars([true_car])
        features, _ctx = extract(healthy)
        order, margin_without = _rank(features, healthy.header_cars, model)

        result = fuse(features, group_weights=model.get("group_weights"),
                      feature_weights=model.get("feature_weights"))
        rows.append({
            "file_id": file_id,
            "true_car": true_car,
            "margin_with_faulty": margin_with,
            "top_score_with_faulty": float(fuse(
                case_set.features[file_id],
                group_weights=model.get("group_weights"),
                feature_weights=model.get("feature_weights"))["score"].max()),
            "healthy_top_car": order[0],
            "healthy_top_score": float(result["score"].max()),
            "margin_healthy_only": margin_without,
            "margin_ratio": (margin_with / margin_without
                             if margin_without and np.isfinite(margin_without) else np.nan),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# C. synthetic recovery / detection limit
# --------------------------------------------------------------------------
DELTAS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00)


def detection_limit(case_set, model: dict, trials_per_delta: int,
                    seed: int = 20260919) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = []
    for file_id in case_set.labelled():
        case = case_set.cases[file_id]
        true_car = case_set.labels[file_id]
        healthy = case.without_cars([true_car])
        if len(healthy.cars) < 3:
            continue
        targets = list(rng.permutation(healthy.cars))[:trials_per_delta]
        for delta in DELTAS:
            for target in targets:
                injected = healthy.with_injected_deficit(target, delta)
                features, _ctx = extract(injected)
                order, margin = _rank(features, injected.header_cars, model)
                rows.append({
                    "file_id": file_id, "delta_k": delta, "target": target,
                    "rank": order.index(target) + 1 if target in order else -1,
                    "score": rank_decay_score(order, target),
                    "margin": margin,
                    "observed_elev": float(features.loc[target, "elev_mean"])
                    if target in features.index else np.nan,
                })
        print(f"  {file_id}: {len(targets)} target cars x {len(DELTAS)} magnitudes")
    long = pd.DataFrame(rows)
    if long.empty:
        return long, long
    summary = (long.groupby("delta_k")
               .agg(trials=("score", "size"),
                    top1_rate=("rank", lambda s: float((s == 1).mean())),
                    top2_rate=("rank", lambda s: float((s <= 2).mean())),
                    mean_score=("score", "mean"),
                    mean_rank=("rank", "mean"),
                    mean_observed_elev=("observed_elev", "mean"))
               .reset_index())
    return summary, long


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=300, help="random weight vectors for test A")
    ap.add_argument("--trials", type=int, default=3, help="target cars per magnitude for test C")
    args = ap.parse_args()

    os.makedirs(cfg.REPORT_DIR, exist_ok=True)
    case_set = build_case_set(verbose=True)
    model = dict(DEFAULT_MODEL)
    out = ["# ACV2 robustness report", "",
           "Three tests aimed at the known weaknesses of a six-file problem. See the "
           "module docstring of `scripts/validate_robustness.py` for what each one can "
           "and cannot establish.", ""]

    print("\n[A] weight sensitivity")
    draws = weight_sensitivity(case_set, args.draws)
    perfect = float((draws["mean_score"] == 1.0).mean())
    quantiles = draws["mean_score"].quantile([0.0, 0.05, 0.25, 0.5]).to_dict()
    print(f"  mean score across draws = {draws['mean_score'].mean():.4f}")
    print(f"  fraction of random weightings scoring a perfect 1.0000 = {perfect:.3f}")
    print(f"  worst / 5th pct / median = {quantiles[0.0]:.4f} / "
          f"{quantiles[0.05]:.4f} / {quantiles[0.5]:.4f}")
    out += ["## A. Weight sensitivity", "",
            f"{len(draws)} random non-negative weight vectors (uniform on the simplex) over the "
            f"same channels, physics-fixed signs retained, all group multipliers 1.0.", "",
            df_to_md(pd.DataFrame([{
                "draws": len(draws),
                "mean_score": draws["mean_score"].mean(),
                "worst": quantiles[0.0],
                "p05": quantiles[0.05],
                "p25": quantiles[0.25],
                "median": quantiles[0.5],
                "fraction_perfect": perfect,
                "hand_tuned_prior": 1.0,
                "random_baseline": RANDOM_BASELINE_8,
            }])), ""]

    print("\n[B] specificity with the faulty car removed")
    spec = specificity(case_set, model)
    print(spec.to_string(index=False))
    ratio = spec["margin_ratio"].median()
    print(f"  median margin ratio (with faulty / healthy only) = {ratio:.2f}x")
    out += ["## B. Specificity: the faulty car deleted", "",
            "Each file re-ranked over its seven healthy siblings only. "
            "`margin_ratio` is the top-1 margin with the faulty car present divided by "
            "the margin among healthy cars alone.", "",
            df_to_md(spec), "",
            f"Median margin ratio **{ratio:.2f}x**.", ""]

    print("\n[C] synthetic recovery and detection limit")
    summary, long = detection_limit(case_set, model, args.trials)
    print(summary.to_string(index=False))
    out += ["## C. Synthetic recovery and detection limit", "",
            "Real faulty car removed, then a physics-shaped capacity deficit of known mean "
            "magnitude injected into one healthy car and re-quantised onto the channel's own "
            "measurement grid. `delta_k` is the mean elevation added, directly comparable to "
            "the `elev_mean` of the real faulty cars (0.18-1.31 K) and of the test case's top "
            "car (0.11 K).", "",
            df_to_md(summary), "",
            "Per-file breakdown:", "",
            df_to_md(long.pivot_table(index="file_id", columns="delta_k",
                                      values="rank", aggfunc="mean"), index=True), ""]

    path = os.path.join(cfg.REPORT_DIR, "robustness.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
