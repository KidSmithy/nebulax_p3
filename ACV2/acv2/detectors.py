"""
acv2.detectors
===========================================================================
Turn the per-car physics features into an anomaly score and a ranking.

The estimation problem is unusual and it dictates the method. There are six
labelled files, one faulty car each - about six useful training examples in
total - so fitting a classifier on the features would overfit immediately.
What *is* abundant is within-file structure: each file contains one faulty car
and seven healthy siblings recorded under identical conditions.

So the detector is a **peer-consensus anomaly detector**, not a classifier:

1. Each feature is converted to a robust z-score against the other cars in the
   same file, using the median and the MAD. The MAD is unaffected by the single
   outlier it is meant to expose, which an ordinary standard deviation is not.
2. Each z-score is oriented by *physics*, not by fitting: the sign of every
   feature is fixed in :data:`acv2.config.FEATURE_SPEC` from the direction the
   energy balance predicts for an undercharged pack.
3. The oriented z-scores are pooled with the physical prior weights,
   renormalised over whichever features the file's schema actually supports.
   A thin-schema file is therefore scored on the thermal evidence alone rather
   than penalised for missing pressure channels.
4. An unsupervised IsolationForest over the same feature space provides a
   second, model-free opinion on which car is the outlier. It is combined with
   a small weight and signed by the physics projection, so a car that is
   anomalously *good* can never be promoted.

Cross validation (scripts/evaluate_loocv.py) is leave-one-file-out and only
ever adjusts group-level multipliers - never a per-feature sign - which keeps
the number of fitted degrees of freedom far below the number of files.
===========================================================================
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from . import config as cfg
from . import physics as phys


# --------------------------------------------------------------------------
# feature -> oriented z-score
# --------------------------------------------------------------------------
def informative_features(features: pd.DataFrame) -> list[str]:
    """Features that can discriminate at all inside this file."""
    usable = []
    for name in cfg.FEATURE_SPEC:
        if name not in features.columns:
            continue
        col = pd.to_numeric(features[name], errors="coerce")
        if col.notna().sum() < 2:
            continue
        if col.nunique(dropna=True) < 2:
            continue                      # constant channel carries no ranking
        usable.append(name)
    return usable


def oriented_z(features: pd.DataFrame) -> pd.DataFrame:
    """Robust z-scores, signed so that a larger value always means more leak-like."""
    names = informative_features(features)
    data = {}
    for name in names:
        orientation = cfg.FEATURE_SPEC[name][1]
        data[name] = orientation * phys.robust_z(pd.to_numeric(features[name], errors="coerce"))
    return pd.DataFrame(data, index=features.index)


# --------------------------------------------------------------------------
# physics score
# --------------------------------------------------------------------------
def physics_score(z: pd.DataFrame,
                  group_weights: dict[str, float] | None = None,
                  feature_weights: dict[str, float] | None = None
                  ) -> tuple[pd.Series, pd.DataFrame, dict]:
    """
    Weighted pool of the oriented z-scores.

    Returns ``(score, contributions, used_weights)``. ``contributions`` is the
    per-car per-feature signed contribution, which is what the explanation in
    :mod:`acv2.ranker` is built from - the diagnosis is always traceable back
    to named physical evidence.
    """
    group_weights = dict(cfg.GROUP_WEIGHTS if group_weights is None else group_weights)
    used: dict[str, float] = {}
    contributions = pd.DataFrame(0.0, index=z.index, columns=z.columns)

    for name in z.columns:
        group, _orientation, default_w, _doc = cfg.FEATURE_SPEC[name]
        weight = default_w if feature_weights is None else feature_weights.get(name, default_w)
        weight *= group_weights.get(group, 1.0)
        if weight <= 0:
            continue
        used[name] = weight
        contributions[name] = weight * z[name]

    total = sum(used.values())
    if total <= 0:
        return pd.Series(0.0, index=z.index), contributions, used
    contributions = contributions / total
    return contributions.sum(axis=1), contributions, used


# --------------------------------------------------------------------------
# unsupervised second opinion
# --------------------------------------------------------------------------
def outlier_score(z: pd.DataFrame, direction: pd.Series) -> pd.Series:
    """
    Signed IsolationForest outlierness over the oriented feature space.

    With only a handful of cars per file this is a weak learner on its own; it
    is included because it is *model free* - it can flag a car whose pattern of
    evidence is jointly unusual even when no single feature stands out - and it
    is signed by the physics projection so it can only ever reinforce or dilute
    a physically sensible verdict.
    """
    if z.shape[0] < 4 or z.shape[1] < 2:
        return pd.Series(0.0, index=z.index)
    matrix = z.to_numpy(dtype=float)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    forest = IsolationForest(
        n_estimators=cfg.FUSION["iso_n_estimators"],
        max_samples=matrix.shape[0],
        random_state=cfg.FUSION["iso_random_state"],
        contamination=min(0.5, max(1.0 / matrix.shape[0], 1e-3)),
    ).fit(matrix)
    # -score_samples is larger for more isolated points
    raw = pd.Series(-forest.score_samples(matrix), index=z.index)
    centred = raw - raw.median()
    scale = phys.mad_scale(centred)
    if not np.isfinite(scale) or scale < 1e-9:
        scale = float(centred.std(ddof=0)) or 1.0
    normalised = (centred / scale).clip(-cfg.PHYSICS["z_clip"], cfg.PHYSICS["z_clip"])
    sign = np.sign(direction.reindex(normalised.index).fillna(0.0))
    return normalised * sign.replace(0.0, 1.0) * (normalised > 0).astype(float)


# --------------------------------------------------------------------------
# fusion
# --------------------------------------------------------------------------
def fuse(features: pd.DataFrame,
         group_weights: dict[str, float] | None = None,
         feature_weights: dict[str, float] | None = None,
         w_physics: float | None = None,
         w_outlier: float | None = None) -> dict:
    """Run the whole detector stack and return every intermediate quantity."""
    w_physics = cfg.FUSION["w_physics"] if w_physics is None else w_physics
    w_outlier = cfg.FUSION["w_outlier"] if w_outlier is None else w_outlier

    z = oriented_z(features)
    if z.empty:
        score = pd.Series(0.0, index=features.index)
        return {"z": z, "physics": score, "outlier": score, "score": score,
                "contributions": pd.DataFrame(index=features.index), "weights": {}}

    physics, contributions, used = physics_score(z, group_weights, feature_weights)
    outlier = outlier_score(z, physics)
    score = w_physics * physics + w_outlier * outlier
    return {
        "z": z,
        "physics": physics,
        "outlier": outlier,
        "score": score,
        "contributions": contributions,
        "weights": used,
    }
