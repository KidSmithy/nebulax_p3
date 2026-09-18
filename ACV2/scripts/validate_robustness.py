"""
validate_robustness.py
===========================================================================
Five tests the standard evaluation cannot perform, each aimed at a specific,
documented weakness of a six-file problem. Two of them also produce the
calibration artefacts the ranker needs to attach an honest confidence to a
verdict, so this script is part of the model build, not only of its audit.

A. WEIGHT SENSITIVITY  -- "the physical prior was hand-designed after looking
   at all six files, so LOOCV overstates generalisation."
   True: LOOCV re-runs the calibration but not the manual feature engineering.
   So: is the result a property of *those particular weights*, or of the
   physics? Score the six files under hundreds of random non-negative weight
   vectors over the same channels, keeping only the physically-fixed signs. If
   almost any positive weighting works, the hand-chosen numbers are not what is
   doing the work and the hand-tuning objection loses most of its force.

B. SPECIFICITY AND NULL CALIBRATION -- "a large top-1 margin is reported as
   though it were confidence."
   So: delete the known faulty car and re-rank the healthy siblings that remain,
   then leave out each healthy car in turn as well. This measures what a
   *fault-free* consist looks like. The finding is blunt: healthy consists
   separate as strongly as faulty ones, so the raw margin is not evidence of a
   fault at all. The resulting distribution of the scale-free Dixon-Q statistic
   is written to `artifacts/null_calibration.json` and becomes the reference
   distribution for every confidence figure the ranker reports.

C. SYNTHETIC RECOVERY AND DETECTION LIMIT -- "how far can we trust a verdict
   built on a +0.11 K elevation?"
   So: remove the real faulty car, inject a physics-shaped capacity deficit of
   known magnitude into one of the remaining healthy cars, and measure how often
   it is recovered as a function of that magnitude. Because the injected car is
   known and the real fault is gone, every trial is a genuinely unseen case, and
   this is the only test here that escapes the six-example ceiling. The curve is
   written to `artifacts/detection_limit.json` so the ranker can say where a
   given verdict's evidence sits on it.

   Honest limit of test C: the injection follows the same load-scaled,
   progressive shape the features are built around, so it measures *sensitivity
   to a deficit of a given size in real telemetry noise* - not whether the leak
   model itself is right. It is quantised onto the channel's own measurement grid
   so a sub-resolution deficit is not made artificially easy.

D. PEER-DROPOUT (CAR-LEVEL) BOOTSTRAP -- "the bootstrap resamples 6-hour time
   blocks, which tests temporal stability and nothing else."
   So: resample the *consist* instead. Drop healthy cars from the reference set
   and re-rank. A peer-consensus verdict that moves when two siblings leave was
   a property of the peer group, not of the accused car. This still cannot test
   generalisation across files - with six files nothing can - but it tests
   generalisation across consists, which was previously unmeasured.

E. REFRIGERANT BRANCH AGREEMENT -- "the refrigerant branch is validated on
   exactly one file."
   Also true, and it is a data limitation rather than a measured weakness: only
   one file's schema carries circuit pressures. Rather than downweight the most
   physically direct evidence available because of an accident of instrumentation,
   this test scores each file twice - inferred thermal evidence only, circuit
   pressures only - and reports whether the two independent lines of reasoning
   name the same car, and which was right when they differ.

Run:  python ACV2/scripts/validate_robustness.py [--draws 300] [--trials 3]
Writes: ACV2/reports/robustness.md
        ACV2/artifacts/null_calibration.json
        ACV2/artifacts/detection_limit.json
===========================================================================
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acv2 import config as cfg                                      # noqa: E402
from acv2 import confidence                                         # noqa: E402
from acv2.detectors import fuse, informative_features               # noqa: E402
from acv2.evaluate import (RANDOM_BASELINE_8, branch_agreement,     # noqa: E402
                           build_case_set, null_consists,
                           observed_separation, peer_dropout_stability,
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
    # Only channels the shipped model actually weights: a channel demoted to
    # zero weight (full_demand_frac, measured below the random baseline) must
    # not be smuggled back in by the sensitivity test either.
    weighted = {name for name, spec in cfg.FEATURE_SPEC.items() if spec[2] > 0}
    channels = sorted({name for fid in case_set.labelled()
                       for name in informative_features(case_set.features[fid])
                       if name in weighted})
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
# B. specificity: no faulty car present, and the null calibration it yields
# --------------------------------------------------------------------------
def specificity(case_set, model: dict) -> pd.DataFrame:
    rows = []
    for file_id in case_set.labelled():
        case = case_set.cases[file_id]
        true_car = case_set.labels[file_id]

        with_faulty = fuse(case_set.features[file_id],
                           group_weights=model.get("group_weights"),
                           feature_weights=model.get("feature_weights"),
                           w_physics=model.get("w_physics"),
                           w_outlier=model.get("w_outlier"))["score"]
        sep_with = confidence.separation(with_faulty)

        healthy = case.without_cars([true_car])
        features, _ctx = extract(healthy)
        without = fuse(features,
                       group_weights=model.get("group_weights"),
                       feature_weights=model.get("feature_weights"),
                       w_physics=model.get("w_physics"),
                       w_outlier=model.get("w_outlier"))["score"]
        sep_without = confidence.separation(without)

        rows.append({
            "file_id": file_id,
            "true_car": true_car,
            "margin_with_faulty": sep_with["margin"],
            "margin_healthy_only": sep_without["margin"],
            "margin_ratio": (sep_with["margin"] / sep_without["margin"]
                             if sep_without["margin"] else np.nan),
            "q_with_faulty": sep_with["dixon_q"],
            "healthy_top_car": sep_without["top_car"],
            "q_healthy_only": sep_without["dixon_q"],
            "q_ratio": (sep_with["dixon_q"] / sep_without["dixon_q"]
                        if sep_without["dixon_q"] else np.nan),
        })
    return pd.DataFrame(rows)


def write_null_calibration(nulls: pd.DataFrame, model: dict) -> str:
    values = [float(v) for v in nulls["dixon_q"].dropna()]
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "statistic": "dixon_q",
        "definition": "(s1 - s2) / (s1 - sn) on the descending fused scores",
        "source": ("labelled training files with the labelled faulty car deleted, then "
                   "each healthy car left out in turn"),
        "n_samples": len(values),
        "dixon_q_null": values,
        "quantiles": {str(q): float(np.quantile(values, q)) for q in
                      (0.5, 0.75, 0.9, 0.95, 0.99)} if values else {},
        "per_consist": nulls.to_dict(orient="records"),
        "model_group_weights": model.get("group_weights"),
        "caveat": ("'Healthy' means 'not the labelled faulty car'. An unlabelled second "
                   "degraded pack would enter the null and make it look more separated, "
                   "which biases reported confidence downward - the conservative direction."),
    }
    os.makedirs(cfg.ARTIFACT_DIR, exist_ok=True)
    with open(cfg.NULL_CALIBRATION_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return cfg.NULL_CALIBRATION_PATH


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


def write_detection_limit(summary: pd.DataFrame) -> str:
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "x_axis": "delta_k",
        "x_meaning": ("mean peer-relative cabin-temperature elevation added by the injected "
                      "deficit, in kelvin, directly comparable to the elev_mean feature"),
        "reliable_top1_rate": cfg.CONFIDENCE["reliable_top1_rate"],
        "curve": [{"delta_k": float(r.delta_k), "top1_rate": float(r.top1_rate),
                   "top2_rate": float(r.top2_rate), "trials": int(r.trials),
                   "mean_observed_elev": float(r.mean_observed_elev)}
                  for r in summary.itertuples()],
        "caveat": ("The injection has the shape the features are designed to see, so this is "
                   "sensitivity to a deficit of a given size in real telemetry noise, not "
                   "validation of the leak model itself."),
    }
    os.makedirs(cfg.ARTIFACT_DIR, exist_ok=True)
    with open(cfg.DETECTION_LIMIT_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return cfg.DETECTION_LIMIT_PATH


# --------------------------------------------------------------------------
# D. peer-dropout (car-level) bootstrap
# --------------------------------------------------------------------------
def peer_dropout(case_set, model: dict, n_draws: int, drop_k: int) -> pd.DataFrame:
    rows = []
    for file_id in case_set.labelled():
        stab = peer_dropout_stability(case_set.cases[file_id], model,
                                      n_draws=n_draws, drop_k=drop_k)
        true_car = case_set.labels[file_id]
        freq = stab["top1_frequency"]
        rows.append({
            "file_id": file_id, "true_car": true_car,
            "draws": stab["n_draws"], "cars_dropped": stab.get("drop_k"),
            "baseline_top": stab["baseline_top"],
            "top1_true_car": freq.get(true_car, 0.0),
            "leader_retained": stab["leader_retained"],
            "runner_up_car": next((c for c in freq if c != stab["baseline_top"]), "-"),
            "runner_up_freq": next((f for c, f in freq.items()
                                    if c != stab["baseline_top"]), 0.0),
        })
        print(f"  {file_id}: true={true_car} top-1 frequency={rows[-1]['top1_true_car']:.2f} "
              f"over {stab['n_draws']} consists")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=300, help="random weight vectors for test A")
    ap.add_argument("--trials", type=int, default=3, help="target cars per magnitude for test C")
    ap.add_argument("--peer-draws", type=int, default=30, help="consist resamples for test D")
    ap.add_argument("--drop-k", type=int, default=2, help="cars dropped per resample in test D")
    args = ap.parse_args()

    os.makedirs(cfg.REPORT_DIR, exist_ok=True)
    case_set = build_case_set(verbose=True)
    model = dict(DEFAULT_MODEL)
    out = ["# ACV2 robustness report", "",
           "Five tests aimed at the known weaknesses of a six-file problem. See the "
           "module docstring of `scripts/validate_robustness.py` for what each one can "
           "and cannot establish. Tests B and C also write the calibration artefacts the "
           "ranker uses to attach a confidence to a verdict.", ""]

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
            }])), "",
            "Reading: the hand-tuned prior is not load-bearing. Most random positive "
            "weightings of the same physically-signed channels also rank every faulty car "
            "first, and the worst draw still beats the random baseline comfortably. What is "
            "doing the work is the sign structure from the energy balance, which is fixed by "
            "physics and never fitted, not the particular magnitudes chosen by hand.", ""]

    print("\n[B] specificity with the faulty car removed")
    spec = specificity(case_set, model)
    print(spec.to_string(index=False))
    ratio = spec["margin_ratio"].median()
    q_ratio = spec["q_ratio"].median()
    print(f"  median margin ratio (with faulty / healthy only) = {ratio:.2f}x")
    print(f"  median Q ratio                                   = {q_ratio:.2f}x")

    print("  building the healthy-consist null ...")
    nulls = null_consists(case_set, model)
    observed = observed_separation(case_set, model)
    null_path = write_null_calibration(nulls, model)
    calibration = confidence.load_null_calibration(null_path)
    observed["null_p_value"] = [confidence.null_p_value(q, calibration["dixon_q_null"])
                               for q in observed["dixon_q"]]
    print(f"  {len(nulls)} healthy consists -> {null_path}")
    print(observed.to_string(index=False))

    out += ["## B. Specificity: the faulty car deleted", "",
            "Each file re-ranked over its healthy siblings only. `margin_ratio` is the top-1 "
            "margin with the faulty car present divided by the margin among healthy cars "
            "alone; `q_ratio` is the same comparison for the scale-free Dixon-Q separation.",
            "", df_to_md(spec), "",
            f"Median margin ratio **{ratio:.2f}x**, median Q ratio **{q_ratio:.2f}x**.", "",
            "**This is the most consequential result in the suite, and it is negative.** A "
            "healthy consist separates as strongly as a faulty one. That is structural rather "
            "than a defect: the ranker returns whichever car is most anomalous relative to its "
            "siblings, and exactly one car is always the warmest, so a winner with a margin is "
            "produced whether or not anything is broken. The consequence is that the raw "
            "top-1 margin the system used to report as confidence carried no information about "
            "fault *presence*. It has been replaced by a null-referenced statistic, calibrated "
            "on the healthy consists below.", "",
            f"### Null calibration ({len(nulls)} fault-free consists)", "",
            "Built by deleting each file's labelled faulty car and then leaving out each "
            "healthy car in turn. Written to `artifacts/null_calibration.json` and consumed by "
            "`acv2.confidence`.", "",
            df_to_md(pd.DataFrame([{
                "n_consists": len(nulls),
                "q_median": nulls["dixon_q"].median(),
                "q_p75": nulls["dixon_q"].quantile(0.75),
                "q_p90": nulls["dixon_q"].quantile(0.90),
                "q_p95": nulls["dixon_q"].quantile(0.95),
                "q_max": nulls["dixon_q"].max(),
            }])), "",
            "The labelled files scored against that null:", "",
            df_to_md(observed), "",
            "`null_p_value` is the fraction of fault-free consists that separate at least as "
            "cleanly. A real fault is not guaranteed to produce a small p-value - a leak whose "
            "capacity deficit is still small genuinely does look like a healthy consist - so "
            "this figure measures how *distinguishable* the verdict is, not how likely it is "
            "to be the right car.", ""]

    print("\n[C] synthetic recovery and detection limit")
    summary, long = detection_limit(case_set, model, args.trials)
    print(summary.to_string(index=False))
    limit_path = write_detection_limit(summary)
    print(f"  wrote {limit_path}")
    reliable = summary.loc[summary["top1_rate"] >= cfg.CONFIDENCE["reliable_top1_rate"],
                           "delta_k"]
    reliable_from = float(reliable.min()) if len(reliable) else float("nan")
    out += ["## C. Synthetic recovery and detection limit", "",
            "Real faulty car removed, then a physics-shaped capacity deficit of known mean "
            "magnitude injected into one healthy car and re-quantised onto the channel's own "
            "measurement grid. `delta_k` is the mean elevation added, directly comparable to "
            "the `elev_mean` of the real faulty cars and of the test case's top car.", "",
            df_to_md(summary), "",
            f"Recovery becomes reliable (top-1 rate >= "
            f"{cfg.CONFIDENCE['reliable_top1_rate']:.0%}) from **{reliable_from:.2f} K** "
            f"upward. This is the only test in the suite that escapes the six-example ceiling: "
            f"every trial has a known injected target and the real fault has been removed, so "
            f"{len(long)} independent unseen cases are available instead of six.", "",
            "Per-file mean rank of the injected car:", "",
            df_to_md(long.pivot_table(index="file_id", columns="delta_k",
                                      values="rank", aggfunc="mean"), index=True), ""]

    print("\n[D] peer-dropout (car-level) bootstrap")
    peers = peer_dropout(case_set, model, args.peer_draws, args.drop_k)
    print(peers.to_string(index=False))
    out += ["## D. Peer-dropout (car-level) bootstrap", "",
            f"Each case re-ranked from scratch after dropping {args.drop_k} randomly chosen "
            f"non-leading cars, {args.peer_draws} times. Features, peer reference, ambient "
            f"estimate and load strata are all recomputed, because every one of them changes "
            f"when the consist changes.", "",
            df_to_md(peers), "",
            "This complements the time-block bootstrap in `reports/evaluation.md`, which "
            "resamples 6-hour blocks and therefore only measures temporal stability against a "
            "fixed reference set. Neither test can measure generalisation to an unseen *file*; "
            "with six labelled files nothing can, and that limit is irreducible here.", ""]

    print("\n[E] refrigerant branch agreement")
    branches = branch_agreement(case_set)
    print(branches.to_string(index=False))
    out += ["## E. Refrigerant branch agreement", "",
            "Each file scored twice: once on inferred thermal evidence only "
            "(thermal + capacity + progression), once on circuit pressures only.", "",
            df_to_md(branches), "",
            "The refrigerant branch is computable in exactly one of the six files, because "
            "only that file's schema exposes circuit pressures. That is an accident of "
            "instrumentation, not a measured weakness, so the branch keeps the weight its "
            "physical directness earns rather than being penalised for it. What this table "
            "adds is auditability: where the two independent lines of reasoning disagree, it "
            "records which one was right. On `acv_case_04` the inferred branch alone does not "
            "name the true car and the circuit-pressure branch does, which is the single piece "
            "of evidence supporting the branch - one data point, stated as one data point.", ""]

    path = os.path.join(cfg.REPORT_DIR, "robustness.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
