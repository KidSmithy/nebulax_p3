"""
acv2.confidence
===========================================================================
How much should the verdict be believed?

The problem this module exists to solve
---------------------------------------
Until now the only confidence signal the system reported was the raw top-1
margin, ``s1 - s2``. That number is not interpretable, and the robustness suite
proved it. Test B in ``scripts/validate_robustness.py`` deletes the known
faulty car from each labelled file and re-ranks the seven healthy siblings that
remain. The top-1 margin among those healthy cars is *the same size* as the
margin with the real fault present - median ratio 0.94x, and on
``acv_case_04`` the healthy-only consist separates more strongly (1.00) than
the genuine fault does (0.19).

The reason is structural, not a bug. The ranker is a peer-consensus detector:
it returns whichever car is most anomalous relative to its siblings. Exactly
one car is always the warmest, so the detector always produces a winner with a
margin, whether or not anything is broken. A margin is evidence about *spread*,
not about *fault presence*.

So confidence here is built from two measured quantities instead:

1. **Separation.** A Dixon-Q statistic, ``(s1 - s2) / (s1 - sn)``. It is
   scale-free - immune to the arbitrary units of a weighted z-score pool, which
   the raw margin is not - bounded in [0, 1], and is the classical small-sample
   test for a single outlier, which is precisely the structure of this problem
   (one faulty car among n siblings).

2. **A null reference.** The same statistic computed on consists that are known
   to contain no fault, obtained by deleting the labelled faulty car from each
   training file and then also leaving out each healthy car in turn. The
   reported p-value is ``P(Q_null >= Q_observed)``: the fraction of genuinely
   healthy consists that look at least as separated as the case in hand.

3. **A detection-limit cross-check.** Test C measures how often a deficit of a
   given magnitude is actually recovered. A verdict whose leading thermal
   evidence is smaller than the magnitude at which recovery becomes reliable is
   reported as provisional however clean its separation looks, because in that
   regime recovery is close to a coin flip.

The honest consequence is that some verdicts - including the shipped test-case
verdict - come back labelled as weakly separated. That is the correct output
for this evidence, and it is more useful to an operator than a confident number
that measurement does not support.
===========================================================================
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import config as cfg
from . import physics as phys


# --------------------------------------------------------------------------
# separation statistics
# --------------------------------------------------------------------------
def dixon_q(scores: pd.Series | np.ndarray) -> float:
    """
    ``(s1 - s2) / (s1 - sn)`` on the descending-sorted scores.

    Scale-free, so it is comparable between a thin-schema file scored on five
    thermal channels and a rich-schema file scored on twelve, which the raw
    margin is not. Returns NaN for fewer than three scores, where the statistic
    is not defined.
    """
    arr = np.sort(np.asarray(scores, dtype=float))[::-1]
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return float("nan")
    spread = arr[0] - arr[-1]
    if spread <= 1e-12:
        return float("nan")
    return float((arr[0] - arr[1]) / spread)


def top_robust_z(scores: pd.Series | np.ndarray) -> float:
    """
    How many robust sigmas the leader sits above its peers, using the median and
    MAD of the *other* cars so the leader cannot inflate its own reference.
    """
    arr = np.asarray(scores, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return float("nan")
    order = np.sort(arr)[::-1]
    leader, others = order[0], order[1:]
    scale = phys.mad_scale(others)
    if not np.isfinite(scale) or scale < 1e-9:
        scale = float(np.std(others, ddof=0))
    if not np.isfinite(scale) or scale < 1e-9:
        return float("nan")
    return float((leader - float(np.median(others))) / scale)


def separation(scores: pd.Series) -> dict:
    """Every scale-free description of "how alone is the leader?" in one dict."""
    ordered = pd.to_numeric(pd.Series(scores), errors="coerce").dropna().sort_values(
        ascending=False)
    if ordered.empty:
        return {"n_cars": 0, "margin": float("nan"), "dixon_q": float("nan"),
                "top_z": float("nan"), "top_car": None, "runner_up": None}
    return {
        "n_cars": int(ordered.size),
        "top_car": str(ordered.index[0]),
        "runner_up": str(ordered.index[1]) if ordered.size > 1 else None,
        "margin": float(ordered.iloc[0] - ordered.iloc[1]) if ordered.size > 1 else float("nan"),
        "dixon_q": dixon_q(ordered),
        "top_z": top_robust_z(ordered),
    }


# --------------------------------------------------------------------------
# null calibration artefact
# --------------------------------------------------------------------------
def load_null_calibration(path: str | None = None) -> dict | None:
    """
    Load the healthy-consist null distribution, or ``None`` if it has not been
    built yet. Absence is not an error: the ranker then reports the separation
    statistic without a probability attached rather than inventing one.
    """
    path = cfg.NULL_CALIBRATION_PATH if path is None else path
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:                                    # pragma: no cover
        return None
    if not payload.get("dixon_q_null"):
        return None
    return payload


def load_detection_limit(path: str | None = None) -> dict | None:
    """Load the measured recovery-versus-magnitude curve, or ``None``."""
    path = cfg.DETECTION_LIMIT_PATH if path is None else path
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:                                    # pragma: no cover
        return None
    return payload if payload.get("curve") else None


def null_p_value(observed: float, null_values: list[float]) -> float:
    """
    ``P(Q_null >= Q_observed)`` with the usual +1 correction, so a finite null
    sample can never report an impossible p-value of exactly zero.
    """
    values = [v for v in (null_values or []) if np.isfinite(v)]
    if not np.isfinite(observed) or not values:
        return float("nan")
    at_least = sum(1 for v in values if v >= observed)
    return float((at_least + 1) / (len(values) + 1))


# --------------------------------------------------------------------------
# combining the two separation statistics
# --------------------------------------------------------------------------
# Q and top_z answer different questions and measurement shows they are
# complementary rather than redundant. Scored against the fault-free null, the
# six labelled faults come out as:
#
#     file          p via Q    p via top_z
#     acv_case_01     0.214        0.048
#     acv_case_02     0.786        0.071
#     acv_case_03     0.071        0.476
#     acv_case_04     0.952        0.762
#     acv_case_05     0.524        0.619
#     acv_case_06     0.095        0.333
#
# Q catches cases 03 and 06; top_z catches 01 and 02. Neither dominates, because
# they fail in different regimes:
#
#   * Q = (s1-s2)/(s1-sn) is a *ratio of gaps*, so it is blind to how far out the
#     leader actually is. It also saturates once the leader's z-scores hit the
#     z_clip guard, since further degradation cannot widen the numerator while
#     the healthy cars' own scatter still fills the denominator.
#   * top_z measures the leader's distance in robust sigmas, so it does not
#     saturate - but it says nothing about whether the *runner-up* is nearly as
#     bad, which is exactly the ambiguity Q is good at exposing.
#
# Using only one therefore discards real evidence. Both are tested and the
# smaller p-value is taken, with a Bonferroni factor of two to pay honestly for
# having looked twice. That is deliberately conservative: it cannot manufacture
# significance, and it lifts four of the six genuine faults to a distinguishable
# p-value where Q alone lifted two.
N_SEPARATION_TESTS = 2


def combined_p_value(p_q: float, p_top_z: float) -> float:
    """Bonferroni-corrected smaller of the two null-referenced p-values."""
    candidates = [p for p in (p_q, p_top_z) if np.isfinite(p)]
    if not candidates:
        return float("nan")
    return float(min(1.0, N_SEPARATION_TESTS * min(candidates)))


def p_value_for(sep: dict, null: dict | None) -> float:
    """
    Combined p-value for an already-computed :func:`separation` dict.

    Convenience for the reporting scripts, so a report can never quote a
    different statistic from the one the ranker acts on.
    """
    if not null:
        return float("nan")
    return combined_p_value(
        null_p_value(sep.get("dixon_q", float("nan")), null.get("dixon_q_null", [])),
        null_p_value(sep.get("top_z", float("nan")), null.get("top_z_null", [])),
    )


# --------------------------------------------------------------------------
# detection-limit cross-check
# --------------------------------------------------------------------------
def detection_context(evidence_k: float, limit: dict | None) -> dict:
    """
    Place the leading thermal evidence on the measured recovery curve.

    ``evidence_k`` is the top car's ``elev_mean`` in kelvin, which is directly
    comparable to the injected ``delta_k`` of test C because the injection is
    normalised to add exactly that mean elevation and is re-quantised onto the
    channel's own measurement grid.
    """
    out = {"evidence_k": float(evidence_k) if np.isfinite(evidence_k) else float("nan"),
           "expected_top1_rate": float("nan"), "reliable_from_k": float("nan"),
           "below_detection_limit": None, "n_trials": 0}
    if limit is None or not np.isfinite(out["evidence_k"]):
        return out
    curve = sorted(({"delta_k": float(p["delta_k"]),
                     "top1_rate": float(p["top1_rate"]),
                     "trials": int(p.get("trials", 0))} for p in limit["curve"]),
                   key=lambda p: p["delta_k"])
    if not curve:
        return out
    reliable = [p["delta_k"] for p in curve
                if p["top1_rate"] >= cfg.CONFIDENCE["reliable_top1_rate"]]
    out["reliable_from_k"] = float(min(reliable)) if reliable else float("nan")
    out["n_trials"] = int(sum(p["trials"] for p in curve))
    # Nearest measured magnitude at or below the observed evidence; if the
    # evidence is smaller than every tested magnitude, use the smallest.
    at_or_below = [p for p in curve if p["delta_k"] <= abs(out["evidence_k"])]
    point = at_or_below[-1] if at_or_below else curve[0]
    out["expected_top1_rate"] = point["top1_rate"]
    out["nearest_tested_k"] = point["delta_k"]
    if np.isfinite(out["reliable_from_k"]):
        out["below_detection_limit"] = bool(abs(out["evidence_k"]) < out["reliable_from_k"])
    return out


# --------------------------------------------------------------------------
# public assessment
# --------------------------------------------------------------------------
BANDS = {
    "strong": "The leader is separated from its siblings more cleanly than any healthy "
              "consist in the calibration set.",
    "moderate": "The leader is separated more cleanly than most healthy consists, but the "
                "pattern is not rare among them.",
    "weak": "A healthy consist of this size routinely separates this cleanly, so the "
            "separation itself is not evidence that a fault is present.",
    "uncalibrated": "No healthy-consist null is available, so the separation cannot be "
                    "turned into a probability.",
}


def assess(scores: pd.Series, evidence_k: float = float("nan"),
           null: dict | None = None, limit: dict | None = None) -> dict:
    """
    Full confidence assessment of one ranking.

    Returns the separation statistics, the null-referenced p-value, the
    detection-limit context and a single ``level`` in
    ``{strong, moderate, weak, uncalibrated}``. The level is deliberately
    downgraded to at most ``moderate`` when the leading evidence falls below the
    measured detection limit, because separation and detectability are different
    questions and a verdict needs both.
    """
    null = load_null_calibration() if null is None else null
    limit = load_detection_limit() if limit is None else limit

    sep = separation(scores)
    null_q = list(null.get("dixon_q_null", [])) if null else []
    null_z = list(null.get("top_z_null", [])) if null else []
    p_q = null_p_value(sep["dixon_q"], null_q)
    p_z = null_p_value(sep["top_z"], null_z)
    p = combined_p_value(p_q, p_z)
    context = detection_context(evidence_k, limit)

    if not np.isfinite(p):
        level = "uncalibrated"
    elif p <= cfg.CONFIDENCE["p_strong"]:
        level = "strong"
    elif p <= cfg.CONFIDENCE["p_moderate"]:
        level = "moderate"
    else:
        level = "weak"

    downgraded_for = None
    if context["below_detection_limit"] and level == "strong":
        level, downgraded_for = "moderate", "below_detection_limit"

    n_null = max(len(null_q), len(null_z))
    thin = bool(n_null) and n_null < cfg.CONFIDENCE["min_null_samples"]
    return {
        **sep,
        "null_p_value": p,
        "null_p_dixon_q": p_q,
        "null_p_top_z": p_z,
        "n_null_samples": n_null,
        "null_thin": thin,
        "null_median_q": float(np.median(null_q)) if null_q else float("nan"),
        "null_median_top_z": float(np.median(null_z)) if null_z else float("nan"),
        "level": level,
        "downgraded_for": downgraded_for,
        "detection": context,
        "statement": BANDS[level],
    }


def describe(assessment: dict) -> str:
    """One paragraph an operator can act on, with every caveat measured."""
    bits = []
    q, z = assessment.get("dixon_q"), assessment.get("top_z")
    p = assessment.get("null_p_value")
    if np.isfinite(q) or np.isfinite(z):
        parts = []
        if np.isfinite(q):
            parts.append(f"Q={q:.2f}")
        if np.isfinite(z):
            parts.append(f"{z:.1f} robust sigma above its peers")
        bits.append("Separation " + " and ".join(parts))
        if np.isfinite(p):
            bits[-1] += (f", which {int(round(p * 100))}% of known-healthy consists match or "
                         f"exceed (n={assessment['n_null_samples']} null consists, two "
                         f"statistics tested)")
        bits[-1] += f". Confidence: {assessment['level'].upper()}."
        bits.append(assessment["statement"])
    det = assessment.get("detection", {})
    if np.isfinite(det.get("expected_top1_rate", float("nan"))):
        bits.append(
            f"Its leading thermal evidence is {det['evidence_k']:+.2f} K; in synthetic-recovery "
            f"trials on this dataset a deficit of that magnitude is recovered as top-1 in "
            f"{det['expected_top1_rate'] * 100:.0f}% of trials, and recovery only becomes "
            f"reliable from {det['reliable_from_k']:.2f} K upward.")
    if assessment.get("downgraded_for") == "below_detection_limit":
        bits.append("Confidence was therefore downgraded: the separation is clean but the "
                    "underlying deficit is near the resolution of the evidence.")
    if assessment.get("null_thin"):
        bits.append(f"The null has only {assessment['n_null_samples']} samples, so the "
                    f"probability is indicative rather than precise.")
    return " ".join(bits)
