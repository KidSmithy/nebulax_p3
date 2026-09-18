"""
acv2.evaluate
===========================================================================
Scoring, diagnostics and honest cross validation.

The competition metric is a linear rank-decay score per file,

    score = (n - (r - 1)) / n ,

where ``r`` is the rank the true faulty car received out of ``n`` ranked cars,
and 0 if it was not ranked at all. A random permutation of 8 cars scores
(n + 1) / (2n) = 0.5625 on average, which is the baseline every result here is
reported against - "better than chance" is a low bar and should be stated
explicitly rather than implied.

With six labelled files, any weight tuning is at serious risk of fitting the
validation set. Two guards are applied:

* calibration only adjusts six group-level multipliers on a coarse grid, never
  a per-feature sign or a per-file quantity;
* the reported number is leave-one-file-out - the weights are re-selected from
  scratch on the other five files for each held-out file, so the held-out score
  never sees its own optimisation.
===========================================================================
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import config as cfg
from .cleaning import CleanCase, clean
from .detectors import fuse, oriented_z
from .features import extract
from .io_loader import discover_cases, load_case, load_labels
from .ranker import DEFAULT_MODEL, rank_case

RANDOM_BASELINE_8 = 0.5625


# --------------------------------------------------------------------------
# metric
# --------------------------------------------------------------------------
def rank_decay_score(ranked: list[str], true_car: str) -> float:
    n = len(ranked)
    if n == 0 or true_car not in ranked:
        return 0.0
    r = ranked.index(true_car) + 1
    return (n - (r - 1)) / n


def rank_of(ranked: list[str], true_car: str) -> int:
    return ranked.index(true_car) + 1 if true_car in ranked else -1


# --------------------------------------------------------------------------
# cached case loading
# --------------------------------------------------------------------------
class CaseSet:
    """Loads, cleans and feature-extracts every labelled case exactly once."""

    def __init__(self, paths: list[str], labels: dict[str, str], verbose: bool = True):
        self.labels = labels
        self.cases: dict[str, CleanCase] = {}
        self.features: dict[str, pd.DataFrame] = {}
        self.contexts: dict[str, dict] = {}
        for path in paths:
            panel = load_case(path, verbose=verbose)
            case = clean(panel)
            features, ctx = extract(case)
            self.cases[panel.file_id] = case
            self.features[panel.file_id] = features
            self.contexts[panel.file_id] = ctx
            if verbose:
                print("  " + case.summary())

    @property
    def file_ids(self) -> list[str]:
        return list(self.cases)

    def labelled(self) -> list[str]:
        return [f for f in self.cases if f in self.labels]

    def header_cars(self, file_id: str) -> list[str]:
        return self.cases[file_id].header_cars

    def ranked(self, file_id: str, model: dict) -> list[str]:
        features = self.features[file_id]
        result = fuse(features,
                      group_weights=model.get("group_weights"),
                      feature_weights=model.get("feature_weights"),
                      w_physics=model.get("w_physics"),
                      w_outlier=model.get("w_outlier"))
        order = result["score"].sort_values(ascending=False).index.tolist()
        tail = [c for c in self.header_cars(file_id) if c not in order]
        return order + tail

    def score(self, model: dict, file_ids: list[str] | None = None) -> pd.DataFrame:
        rows = []
        for file_id in (file_ids or self.labelled()):
            ranked = self.ranked(file_id, model)
            true_car = self.labels[file_id]
            rows.append({
                "file_id": file_id,
                "true_car": true_car,
                "rank": rank_of(ranked, true_car),
                "n_ranked": len(ranked),
                "score": rank_decay_score(ranked, true_car),
                "ranked_cars": "|".join(ranked),
            })
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# per-feature diagnostics
# --------------------------------------------------------------------------
def per_feature_ranks(case_set: CaseSet) -> pd.DataFrame:
    """
    For every feature, where would the true faulty car rank if that feature were
    used alone? This is the sanity check that each physical channel is doing
    real work, and it exposes features that are pure noise or inverted.
    """
    rows = []
    for file_id in case_set.labelled():
        features = case_set.features[file_id]
        z = oriented_z(features)
        true_car = case_set.labels[file_id]
        for name in z.columns:
            order = z[name].sort_values(ascending=False).index.tolist()
            tail = [c for c in case_set.header_cars(file_id) if c not in order]
            ranked = order + tail
            rows.append({
                "feature": name,
                "group": cfg.FEATURE_SPEC[name][0],
                "file_id": file_id,
                "rank": rank_of(ranked, true_car),
                "score": rank_decay_score(ranked, true_car),
                "z_true": float(z[name].get(true_car, np.nan)),
                "z_max_other": float(z[name].drop(index=[true_car], errors="ignore").max()),
            })
    long = pd.DataFrame(rows)
    if long.empty:
        return long
    summary = (long.groupby(["group", "feature"])
               .agg(mean_score=("score", "mean"),
                    n_files=("score", "size"),
                    n_top1=("rank", lambda s: int((s == 1).sum())),
                    mean_rank=("rank", "mean"),
                    mean_margin=("z_true", lambda s: float(np.nanmean(s))))
               .reset_index()
               .sort_values(["group", "mean_score"], ascending=[True, False]))
    return summary, long


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------
GROUP_GRID = (0.0, 0.5, 1.0, 1.5, 2.0)


def _model_with_groups(groups: dict[str, float]) -> dict:
    model = dict(DEFAULT_MODEL)
    model["group_weights"] = dict(groups)
    return model


def calibrate_groups(case_set: CaseSet, train_files: list[str],
                     passes: int = 2, verbose: bool = False) -> tuple[dict, float]:
    """
    Coordinate ascent on the six group multipliers over a coarse grid.

    Ties are broken toward the physical prior, so when a weight change makes no
    measurable difference the model stays where physics put it rather than
    drifting to an arbitrary optimum of six data points.
    """
    groups = dict(cfg.GROUP_WEIGHTS)
    best = float(case_set.score(_model_with_groups(groups), train_files)["score"].mean())
    for _ in range(passes):
        improved = False
        for group in cfg.FEATURE_GROUPS:
            current = groups[group]
            for candidate in GROUP_GRID:
                if candidate == current:
                    continue
                trial = dict(groups)
                trial[group] = candidate
                value = float(case_set.score(_model_with_groups(trial), train_files)["score"].mean())
                if value > best + 1e-9:
                    best, groups, improved = value, trial, True
                    if verbose:
                        print(f"    {group} -> {candidate}: train score {value:.4f}")
        if not improved:
            break
    return groups, best


def loocv(case_set: CaseSet, verbose: bool = True) -> pd.DataFrame:
    """
    Leave-one-file-out evaluation of the *whole procedure*, calibration included.

    For each held-out file the group weights are re-selected using only the
    other files, so the reported score is not contaminated by having seen the
    answer.
    """
    files = case_set.labelled()
    rows = []
    for held_out in files:
        train_files = [f for f in files if f != held_out]
        groups, train_score = calibrate_groups(case_set, train_files)
        model = _model_with_groups(groups)
        scored = case_set.score(model, [held_out]).iloc[0]
        rows.append({
            "held_out": held_out,
            "true_car": scored["true_car"],
            "rank": scored["rank"],
            "score": scored["score"],
            "ranked_cars": scored["ranked_cars"],
            "train_score": train_score,
            "group_weights": groups,
        })
        if verbose:
            print(f"  [loocv] {held_out}: true={scored['true_car']} rank={scored['rank']} "
                  f"score={scored['score']:.3f} (train {train_score:.3f})")
    return pd.DataFrame(rows)


def ablation(case_set: CaseSet, verbose: bool = True) -> pd.DataFrame:
    """Score with each feature group removed, to expose what actually matters."""
    rows = []
    base = float(case_set.score(DEFAULT_MODEL)["score"].mean())
    rows.append({"removed": "(nothing)", "score": base, "delta": 0.0})
    for group in cfg.FEATURE_GROUPS:
        groups = dict(cfg.GROUP_WEIGHTS)
        groups[group] = 0.0
        value = float(case_set.score(_model_with_groups(groups))["score"].mean())
        rows.append({"removed": group, "score": value, "delta": value - base})
    for group in cfg.FEATURE_GROUPS:
        groups = {g: (1.0 if g == group else 0.0) for g in cfg.FEATURE_GROUPS}
        value = float(case_set.score(_model_with_groups(groups))["score"].mean())
        rows.append({"removed": f"all but {group}", "score": value, "delta": value - base})
    table = pd.DataFrame(rows)
    if verbose:
        print(table.to_string(index=False))
    return table


def build_case_set(verbose: bool = True) -> CaseSet:
    paths = discover_cases(cfg.TRAIN_DIR)
    return CaseSet(paths, load_labels(), verbose=verbose)


# --------------------------------------------------------------------------
# stability
# --------------------------------------------------------------------------
def block_bootstrap_stability(case: CleanCase, model: dict | None = None,
                              n_draws: int = 40, block_hours: float = 6.0,
                              keep_fraction: float = 0.7,
                              seed: int = 20260918) -> dict:
    """
    How confident is the verdict? Re-rank the case on random subsets of the
    record and count how often each car comes first.

    Whole blocks of several hours are kept or dropped together rather than
    individual rows, because consecutive 30-second samples are strongly
    autocorrelated and an i.i.d. bootstrap would wildly overstate the effective
    sample size. This is the honest answer to "there are only six cases": a
    single verdict that survives resampling of its own evidence is worth more
    than a point estimate.
    """
    model = dict(DEFAULT_MODEL) if model is None else model
    rng = np.random.default_rng(seed)
    hours = (case.time - case.time.min()).dt.total_seconds() / 3600.0
    block = np.floor(hours / block_hours).astype("Int64")
    block_ids = np.array(sorted(b for b in block.dropna().unique()))
    if len(block_ids) < 4:
        return {"n_draws": 0, "top1_frequency": {}, "mean_rank": {}}

    top1: dict[str, int] = {}
    rank_sum: dict[str, list[int]] = {}
    for _ in range(n_draws):
        chosen = rng.choice(block_ids, size=max(2, int(round(keep_fraction * len(block_ids)))),
                            replace=False)
        mask = block.isin(set(chosen.tolist())).fillna(False)
        try:
            features, _ctx = extract(case.subset(mask))
            result = fuse(features,
                          group_weights=model.get("group_weights"),
                          feature_weights=model.get("feature_weights"),
                          w_physics=model.get("w_physics"),
                          w_outlier=model.get("w_outlier"))
            order = result["score"].sort_values(ascending=False).index.tolist()
        except Exception:
            continue
        if not order:
            continue
        top1[order[0]] = top1.get(order[0], 0) + 1
        for position, car in enumerate(order, start=1):
            rank_sum.setdefault(car, []).append(position)

    draws = sum(top1.values())
    return {
        "n_draws": draws,
        "top1_frequency": {car: count / draws for car, count in
                           sorted(top1.items(), key=lambda kv: -kv[1])} if draws else {},
        "mean_rank": {car: float(np.mean(ranks)) for car, ranks in
                      sorted(rank_sum.items(), key=lambda kv: np.mean(kv[1]))},
    }

