"""
validate_physics_ranker.py
==============================================================================
Sanity check and robustness probe for the ACV physics ranker.

There is no label for the held-out test case, so nothing here "tunes" the
model -- it has no tunable parameters. The purpose is narrower and honest:

  A. Does the physics index put the KNOWN faulty car at or near the top of the
     6 labelled journeys? (Train_Labels.csv is the only ground truth there is.)
  B. Is the test-case answer stable when the fixed weights are perturbed, when
     any single feature is deleted, and when scored day-by-day instead of over
     the whole journey?

If (A) is good and (B) is stable, the ranking is trustworthy for the reason a
physics model is trustworthy: it was never fitted to anything.
==============================================================================
"""

from __future__ import annotations

import os
import itertools
import numpy as np
import pandas as pd

from acv_physics_ranker import (
    FEATURE_ORDER,
    PHYSICS_WEIGHTS,
    rank_consist,
    extract_features,
    robust_z,
    load_case,
    clean_indoor_temperatures,
    cooling_mask,
    setpoints,
    header_car_ids,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "PS3", "02_Datasets", "ACV")
TRAIN_DIR = os.path.join(DATA, "Train")
TEST_FILE = os.path.join(DATA, "Test", "acv_test_case.xlsx")
LABELS = os.path.join(DATA, "Train_Labels.csv")

RULE = "=" * 100


def rank_decay(ranked: list[str], true_car: str) -> tuple[int, float]:
    n = len(ranked)
    if true_car not in ranked:
        return -1, 0.0
    r = ranked.index(true_car) + 1
    return r, (n - (r - 1)) / n


def score_from_features(feats: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    z = pd.DataFrame({f: robust_z(feats[f]) for f in FEATURE_ORDER}, index=feats.index)
    return sum(weights[f] * z[f] for f in FEATURE_ORDER).sort_values(ascending=False)


def main() -> None:
    labels = pd.read_csv(LABELS, dtype=str)
    labels["faulty_car"] = labels["faulty_car"].str.zfill(2)
    truth = dict(zip(labels["filename"], labels["faulty_car"]))

    # ---------------------------------------------------------------- A ----
    print(RULE)
    print("A. LABELLED JOURNEYS -- physics index vs Train_Labels.csv")
    print(RULE)

    cache: dict[str, pd.DataFrame] = {}
    rows = []
    for fname in sorted(truth):
        path = os.path.join(TRAIN_DIR, fname)
        res = rank_consist(path)
        cache[fname] = res["features"]
        r, s = rank_decay(res["ranked_cars_list"], truth[fname])
        rows.append({
            "case": fname.replace("acv_", "").replace(".xlsx", ""),
            "n_ranked": res["consist_size"],
            "n_assessable": len(res["active_cars"]),
            "dt_s": int(res["median_dt_s"]),
            "true": truth[fname],
            "rank": r,
            "score": round(s, 4),
            "margin": res["confidence_margin"],
            "ranked_cars": res["ranked_cars"],
        })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    mean_score = table["score"].mean()
    print(f"\n  mean linear rank-decay score = {mean_score:.4f}")
    print(f"  top-1 hits                   = {(table['rank'] == 1).sum()}/{len(table)}")
    # Expected score of a uniformly random permutation, per case.
    rnd = np.mean([(n + 1) / (2 * n) for n in table["n_ranked"]])
    print(f"  random-permutation baseline  = {rnd:.4f}")

    # ---------------------------------------------------------------- B1 ---
    print()
    print(RULE)
    print("B1. WEIGHT PERTURBATION -- does the answer depend on the chosen weights?")
    print(RULE)
    test_feats = extract_features(TEST_FILE)["features"]
    cache["acv_test_case.xlsx"] = test_feats

    variants: dict[str, dict[str, float]] = {
        "default (1, .5, .25, .25, .25)": dict(PHYSICS_WEIGHTS),
        "mean_rel only":                  {f: (1.0 if f == "mean_rel_cooling" else 0.0) for f in FEATURE_ORDER},
        "all features equal":             {f: 1.0 for f in FEATURE_ORDER},
        "setpoint-heavy (1, 1, .25x3)":   {**PHYSICS_WEIGHTS, "mean_t_minus_set": 1.0},
        "peer-relative only (no setpt)":  {**PHYSICS_WEIGHTS, "mean_t_minus_set": 0.0},
        "persistence-heavy":              {**PHYSICS_WEIGHTS, "persistence_15m": 1.0},
    }

    for label, w in variants.items():
        train_scores = []
        for fname in sorted(truth):
            order = score_from_features(cache[fname], w).index.tolist()
            missing = [c for c in header_car_ids(
                pd.read_excel(os.path.join(TRAIN_DIR, fname), nrows=0).columns) if c not in order]
            _, s = rank_decay(order + missing, truth[fname])
            train_scores.append(s)
        test_order = score_from_features(test_feats, w).index.tolist()
        print(f"  {label:<32} train={np.mean(train_scores):.4f}  test_top={test_order[0]}  "
              f"test_order={'|'.join(test_order)}")

    # ---------------------------------------------------------------- B2 ---
    print()
    print(RULE)
    print("B2. LEAVE-ONE-FEATURE-OUT -- is any single feature carrying the answer alone?")
    print(RULE)
    for drop in FEATURE_ORDER:
        w = {f: (0.0 if f == drop else PHYSICS_WEIGHTS[f]) for f in FEATURE_ORDER}
        train_scores = []
        for fname in sorted(truth):
            order = score_from_features(cache[fname], w).index.tolist()
            missing = [c for c in header_car_ids(
                pd.read_excel(os.path.join(TRAIN_DIR, fname), nrows=0).columns) if c not in order]
            _, s = rank_decay(order + missing, truth[fname])
            train_scores.append(s)
        test_order = score_from_features(test_feats, w).index.tolist()
        print(f"  without {drop:<20} train={np.mean(train_scores):.4f}  "
              f"test_top={test_order[0]}  test_order={'|'.join(test_order)}")

    # ---------------------------------------------------------------- B3 ---
    print()
    print(RULE)
    print("B3. ROBUST (median/MAD) vs CLASSICAL (mean/sigma) standardisation")
    print(RULE)

    def classical_z(v: pd.Series) -> pd.Series:
        sd = float(v.std(ddof=0))
        return (v - v.mean()) / sd if sd > 1e-9 else pd.Series(0.0, index=v.index)

    for name, zfun in (("robust  median/MAD", robust_z), ("classic mean/sigma", classical_z)):
        scores = []
        for fname in sorted(truth):
            f = cache[fname]
            z = pd.DataFrame({c: zfun(f[c]) for c in FEATURE_ORDER}, index=f.index)
            order = sum(PHYSICS_WEIGHTS[c] * z[c] for c in FEATURE_ORDER) \
                .sort_values(ascending=False).index.tolist()
            missing = [c for c in header_car_ids(
                pd.read_excel(os.path.join(TRAIN_DIR, fname), nrows=0).columns) if c not in order]
            _, s = rank_decay(order + missing, truth[fname])
            scores.append(s)
        z = pd.DataFrame({c: zfun(test_feats[c]) for c in FEATURE_ORDER}, index=test_feats.index)
        tscore = sum(PHYSICS_WEIGHTS[c] * z[c] for c in FEATURE_ORDER).sort_values(ascending=False)
        print(f"  {name}: train={np.mean(scores):.4f}  test_order={'|'.join(tscore.index)}  "
              f"margin={tscore.iloc[0] - tscore.iloc[1]:.2f}")

    # ---------------------------------------------------------------- B4 ---
    print()
    print(RULE)
    print("B4. PER-DAY STABILITY -- is the verdict driven by one unusual day?")
    print(RULE)
    for fname in sorted(truth) + ["acv_test_case.xlsx"]:
        path = TEST_FILE if fname.startswith("acv_test") else os.path.join(TRAIN_DIR, fname)
        df, colmap, all_ids, tcol = load_case(path)
        if tcol is None:
            continue
        indoor = clean_indoor_temperatures(df, colmap, all_ids)
        idx = indoor.index
        cool = cooling_mask(df, colmap, all_ids, idx)
        sp = setpoints(df, colmap, all_ids, idx)
        active = [c for c in indoor.columns if indoor[c].notna().any()]
        times = pd.to_datetime(df[tcol], errors="coerce").loc[idx]

        print(f"\n  {fname}  (true fault: {truth.get(fname, '?')})")
        tally = {}
        for day, day_idx in times.groupby(times.dt.date).groups.items():
            ic = indoor.loc[day_idx, active].where(cool.loc[day_idx, active])
            if ic.notna().sum(axis=1).max() < 2:
                continue
            enough = ic.notna().sum(axis=1) >= min(3, len(active))
            rel = ic.sub(ic.median(axis=1), axis=0).where(enough)
            tms = ic.sub(sp.loc[day_idx, active]).where(enough)
            f = pd.DataFrame({
                "mean_rel_cooling": rel.mean(),
                "mean_t_minus_set": tms.mean(),
                "p95_rel": rel.quantile(0.95),
                "frac_rel_elevated": (rel > 0.20).sum() / rel.notna().sum().replace(0, np.nan),
                "persistence_15m": (rel.rolling(30, min_periods=10).mean() > 0.20).sum()
                                   / rel.notna().sum().replace(0, np.nan),
            }).fillna(0.0)
            order = score_from_features(f, PHYSICS_WEIGHTS).index.tolist()
            tr = truth.get(fname)
            rk = order.index(tr) + 1 if tr in order else "-"
            tally[order[0]] = tally.get(order[0], 0) + 1
            print(f"    {day}  n={len(day_idx):5d}  top={order[0]}  true_rank={rk}  "
                  f"order={'|'.join(order)}")
        print(f"    day-level top-1 tally: {tally}")

    print()
    print(RULE)
    print(f"VERDICT: labelled-journey mean score {mean_score:.4f} "
          f"({(table['rank'] == 1).sum()}/{len(table)} exact hits) vs random {rnd:.4f}")
    print(RULE)


if __name__ == "__main__":
    main()
