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
from . import confidence
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


def peer_dropout_stability(case: CleanCase, model: dict | None = None,
                           n_draws: int = 40, drop_k: int = 2,
                           seed: int = 20260919) -> dict:
    """
    Car-level bootstrap: does the verdict survive a change of *reference set*?

    This answers a criticism the block bootstrap cannot. Resampling 6-hour time
    blocks correctly handles autocorrelation, but every draw still judges the
    cars against the same seven siblings, so it measures within-case temporal
    stability and nothing else. A peer-consensus detector has a second and
    arguably more dangerous failure mode: the verdict may be an artefact of
    *which* siblings happen to form the reference median. If dropping two
    healthy cars moves the leader, the diagnosis was a property of the peer
    group rather than of the accused car.

    Cars other than the current leader are dropped, ``drop_k`` at a time, and
    the case is re-ranked from the remaining consist - features recomputed from
    scratch, because the peer reference, the ambient estimate and the load
    strata all change when the consist changes.

    It still cannot test generalisation to an unseen *file*; with six labelled
    files nothing can. It tests generalisation to an unseen *consist*, which is
    a real and previously unmeasured axis.
    """
    model = dict(DEFAULT_MODEL) if model is None else model
    rng = np.random.default_rng(seed)

    baseline_features, _ctx = extract(case)
    baseline = fuse(baseline_features,
                    group_weights=model.get("group_weights"),
                    feature_weights=model.get("feature_weights"),
                    w_physics=model.get("w_physics"),
                    w_outlier=model.get("w_outlier"))["score"].sort_values(ascending=False)
    if baseline.empty:
        return {"n_draws": 0, "baseline_top": None, "top1_frequency": {},
                "leader_retained": float("nan"), "mean_rank": {}}
    leader = str(baseline.index[0])
    candidates = [c for c in case.cars if c != leader]
    drop_k = max(1, min(drop_k, max(0, len(candidates) - 2)))
    if len(candidates) - drop_k < 2:
        return {"n_draws": 0, "baseline_top": leader, "top1_frequency": {},
                "leader_retained": float("nan"), "mean_rank": {}}

    top1: dict[str, int] = {}
    rank_sum: dict[str, list[int]] = {}
    for _ in range(n_draws):
        dropped = rng.choice(candidates, size=drop_k, replace=False).tolist()
        try:
            features, _ = extract(case.without_cars(dropped))
            order = fuse(features,
                         group_weights=model.get("group_weights"),
                         feature_weights=model.get("feature_weights"),
                         w_physics=model.get("w_physics"),
                         w_outlier=model.get("w_outlier")
                         )["score"].sort_values(ascending=False).index.tolist()
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
        "drop_k": drop_k,
        "baseline_top": leader,
        "leader_retained": top1.get(leader, 0) / draws if draws else float("nan"),
        "top1_frequency": {car: count / draws for car, count in
                           sorted(top1.items(), key=lambda kv: -kv[1])} if draws else {},
        "mean_rank": {car: float(np.mean(ranks)) for car, ranks in
                      sorted(rank_sum.items(), key=lambda kv: np.mean(kv[1]))},
    }


# --------------------------------------------------------------------------
# branch agreement
# --------------------------------------------------------------------------
# The refrigerant branch is computable in exactly one of the six labelled files,
# because only that file's schema exposes circuit pressures. That is a data
# limitation, not a measured weakness, and the wrong response is to downweight
# the branch: a direct pressure measurement on the refrigerant loop is
# physically the most informative evidence available, and weight should reflect
# physical directness rather than how many of our six files happened to carry
# pressure sensors.
#
# The right response is to make the branch auditable. This diagnostic scores
# each file twice - once on inferred thermal evidence only, once on circuit
# pressures only - and reports whether the two independent lines of reasoning
# name the same car. Where they disagree it reports which one was right. With
# n=1 that is a single data point and is labelled as such, but it is a stated
# data point instead of a hidden assumption.
INFERRED_GROUPS = ("thermal", "capacity", "progression")


def _branch_model(groups: tuple[str, ...]) -> dict:
    return _model_with_groups({g: (1.0 if g in groups else 0.0) for g in cfg.FEATURE_GROUPS})


def branch_agreement(case_set: CaseSet) -> pd.DataFrame:
    """Per file: top car from the inferred branch vs from the refrigerant branch."""
    rows = []
    for file_id in case_set.labelled():
        features = case_set.features[file_id]
        refrigerant_cols = [n for n, spec in cfg.FEATURE_SPEC.items()
                            if spec[0] == "refrigerant" and n in features.columns
                            and features[n].notna().sum() >= 2
                            and features[n].nunique(dropna=True) > 1]
        if not refrigerant_cols:
            rows.append({"file_id": file_id, "true_car": case_set.labels[file_id],
                         "refrigerant_available": False, "inferred_top": None,
                         "refrigerant_top": None, "agree": None,
                         "inferred_correct": None, "refrigerant_correct": None,
                         "fused_correct": None})
            continue
        true_car = case_set.labels[file_id]
        inferred = case_set.ranked(file_id, _branch_model(INFERRED_GROUPS))
        circuit = case_set.ranked(file_id, _branch_model(("refrigerant",)))
        fused = case_set.ranked(file_id, DEFAULT_MODEL)
        rows.append({
            "file_id": file_id, "true_car": true_car, "refrigerant_available": True,
            "inferred_top": inferred[0], "refrigerant_top": circuit[0],
            "agree": inferred[0] == circuit[0],
            "inferred_correct": inferred[0] == true_car,
            "refrigerant_correct": circuit[0] == true_car,
            "fused_correct": fused[0] == true_car,
            "inferred_rank_true": rank_of(inferred, true_car),
            "refrigerant_rank_true": rank_of(circuit, true_car),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# null calibration: what does a healthy consist look like?
# --------------------------------------------------------------------------
def null_consists(case_set: CaseSet, model: dict | None = None,
                  extra_leave_one_out: bool = True) -> pd.DataFrame:
    """
    Separation statistics for consists that are known to contain no fault.

    Each labelled file has its labelled faulty car deleted, which leaves a
    consist of healthy siblings. Because the separation statistic depends on the
    consist size, each healthy car is then also left out in turn, giving a
    second family of smaller healthy consists and roughly eight null samples per
    file instead of one. These are the reference distribution the ranker's
    p-value is computed against.

    Caveat, stated rather than buried: "healthy" here means "not the labelled
    faulty car". If a file contained a second, unlabelled degraded pack it would
    enter the null and make the null look more separated than it should, which
    biases the reported confidence *downward* - the conservative direction.
    """
    model = dict(DEFAULT_MODEL) if model is None else model
    rows = []
    for file_id in case_set.labelled():
        case = case_set.cases[file_id]
        true_car = case_set.labels[file_id]
        healthy = case.without_cars([true_car])
        variants = [([true_car], healthy)]
        if extra_leave_one_out:
            for car in healthy.cars:
                if len(healthy.cars) - 1 >= 3:
                    variants.append(([true_car, car], case.without_cars([true_car, car])))
        for dropped, consist in variants:
            try:
                features, _ctx = extract(consist)
                scores = fuse(features,
                              group_weights=model.get("group_weights"),
                              feature_weights=model.get("feature_weights"),
                              w_physics=model.get("w_physics"),
                              w_outlier=model.get("w_outlier"))["score"]
            except Exception:
                continue
            sep = confidence.separation(scores)
            if not np.isfinite(sep["dixon_q"]):
                continue
            rows.append({"file_id": file_id, "dropped": "|".join(dropped),
                         "n_cars": sep["n_cars"], "top_car": sep["top_car"],
                         "margin": sep["margin"], "dixon_q": sep["dixon_q"],
                         "top_z": sep["top_z"]})
    return pd.DataFrame(rows)


def observed_separation(case_set: CaseSet, model: dict | None = None) -> pd.DataFrame:
    """Separation statistics for the labelled files as they actually are."""
    model = dict(DEFAULT_MODEL) if model is None else model
    rows = []
    for file_id in case_set.labelled():
        features = case_set.features[file_id]
        scores = fuse(features,
                      group_weights=model.get("group_weights"),
                      feature_weights=model.get("feature_weights"),
                      w_physics=model.get("w_physics"),
                      w_outlier=model.get("w_outlier"))["score"]
        sep = confidence.separation(scores)
        rows.append({"file_id": file_id, "true_car": case_set.labels[file_id],
                     "top_car": sep["top_car"], "correct": sep["top_car"] == case_set.labels[file_id],
                     "n_cars": sep["n_cars"], "margin": sep["margin"],
                     "dixon_q": sep["dixon_q"], "top_z": sep["top_z"],
                     "elev_mean_top": float(features.loc[sep["top_car"], "elev_mean"])
                     if sep["top_car"] in features.index else np.nan})
    return pd.DataFrame(rows)

