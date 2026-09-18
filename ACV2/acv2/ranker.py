"""
acv2.ranker
===========================================================================
End-to-end localisation: one case file in, a ranking of every car out.

The submission format requires *every* car that appears in the file's headers
to be ranked, using the file's own two-digit identifiers. Cars with no usable
telemetry (four of the eight in the rich-schema case) cannot be diagnosed, so
they are appended after every diagnosable car in header order - they are never
interleaved, and a diagnosable car is never pushed below one of them.
===========================================================================
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import config as cfg
from . import confidence
from .cleaning import CleanCase, clean
from .detectors import fuse
from .features import extract
from .io_loader import CasePanel, load_case


# --------------------------------------------------------------------------
# model artefact (weights only - there is no fitted estimator to pickle)
# --------------------------------------------------------------------------
DEFAULT_MODEL = {
    "version": "2.0.0",
    "group_weights": dict(cfg.GROUP_WEIGHTS),
    "feature_weights": {name: spec[2] for name, spec in cfg.FEATURE_SPEC.items()},
    "w_physics": cfg.FUSION["w_physics"],
    "w_outlier": cfg.FUSION["w_outlier"],
    "calibrated": False,
}


def _stale_weights(artefact: dict) -> list[str]:
    """
    Channels whose weight in the artefact disagrees with the current physical
    prior.

    This guard exists because a real bug was possible without it: the artefact's
    ``feature_weights`` are merged *over* the config defaults, so a channel
    demoted in :mod:`acv2.config` would keep its old weight until someone
    remembered to re-run training. A deliberate demotion that silently does
    nothing is worse than no demotion at all, so the mismatch is reported.
    """
    saved = artefact.get("feature_weights") or {}
    return sorted(name for name, spec in cfg.FEATURE_SPEC.items()
                  if name in saved and abs(float(saved[name]) - float(spec[2])) > 1e-9)


def load_model(path: str | None = None) -> dict:
    """
    Load calibrated weights if they exist, otherwise fall back to the physical
    prior. The system is deliberately runnable with no trained artefact at all:
    the physics, not a fitted estimator, is what localises the fault.
    """
    path = cfg.MODEL_PATH if path is None else path
    if path and os.path.exists(path):
        try:
            import joblib
            saved = joblib.load(path)
            model = dict(DEFAULT_MODEL)
            model.update(saved)
            stale = _stale_weights(saved)
            if stale:
                print(f"[acv2] WARNING: {os.path.basename(path)} was built before the current "
                      f"physical prior and disagrees on {', '.join(stale)}; re-run "
                      f"scripts/train.py. Using the config prior for those channels.")
                for name in stale:
                    model["feature_weights"][name] = cfg.FEATURE_SPEC[name][2]
            return model
        except Exception as exc:                      # pragma: no cover
            print(f"[acv2] could not load {path} ({exc}); using physical prior")
    return dict(DEFAULT_MODEL)


# --------------------------------------------------------------------------
# explanation
# --------------------------------------------------------------------------
def _explain(car: str, contributions: pd.DataFrame, features: pd.DataFrame,
             top_n: int = 4) -> list[dict]:
    if car not in contributions.index:
        return []
    row = contributions.loc[car].dropna().sort_values(ascending=False)
    evidence = []
    for name, contribution in row.head(top_n).items():
        if contribution <= 1e-6:
            continue
        group, _orientation, _w, doc = cfg.FEATURE_SPEC[name]
        evidence.append({
            "feature": name,
            "group": group,
            "value": float(features.loc[car, name]) if name in features.columns else float("nan"),
            "contribution": float(contribution),
            "meaning": doc,
        })
    return evidence


def _verdict(top: str, score: pd.Series, features: pd.DataFrame,
             evidence: list[dict], has_refrigerant: bool,
             assessment: dict | None = None) -> str:
    ordered = score.sort_values(ascending=False)
    bits = [f"Car {top} is the most likely refrigerant-leak location "
            f"(score {ordered.iloc[0]:.3f}; runner-up car {ordered.index[1]} "
            f"at {ordered.iloc[1]:.3f})."
            if len(ordered) > 1 else f"Car {top} is the most likely refrigerant-leak location."]
    elev = features.loc[top, "elev_mean"] if "elev_mean" in features.columns else np.nan
    if np.isfinite(elev):
        bits.append(f"Its saloon runs {elev:+.2f} K against the sibling-car median while cooling.")
    if has_refrigerant:
        asym = features.loc[top, "circuit_asym_lift"] if "circuit_asym_lift" in features.columns else np.nan
        exc = features.loc[top, "suction_excursion"] if "suction_excursion" in features.columns else np.nan
        if np.isfinite(asym):
            bits.append(f"Its two refrigeration circuits disagree by {100 * asym:.1f}% in pressure lift")
            if np.isfinite(exc):
                bits[-1] += f", and the weaker circuit's suction pressure dips {100 * exc:.0f}% below its own median"
            bits[-1] += "."
    if evidence:
        bits.append("Leading evidence: " + ", ".join(e["feature"] for e in evidence) + ".")
    # The raw margin is deliberately not quoted as confidence. Test B in
    # scripts/validate_robustness.py showed that healthy consists produce margins
    # of the same size, so quoting one would overstate the evidence.
    if assessment:
        statement = confidence.describe(assessment)
        if statement:
            bits.append(statement)
    return " ".join(bits)


# --------------------------------------------------------------------------
# competing hypothesis
# --------------------------------------------------------------------------
def _contest(top: str, runner_up: str | None, contributions: pd.DataFrame,
             features: pd.DataFrame, z: pd.DataFrame, top_n: int = 5) -> dict:
    """
    Why the runner-up might be right instead.

    A single ranked list hides the shape of the disagreement. When the evidence
    is split across branches - one car warmer and falling further behind set
    point, another reporting invalid telemetry and dropouts - the ranking picks
    a winner but the operator needs to know that a second, physically different
    hypothesis exists and what would settle it. So every weighted channel is
    reported on the side it actually favours, with the margin it contributes.
    """
    if runner_up is None or top not in contributions.index or runner_up not in contributions.index:
        return {"alternative": None, "supports_leader": [], "supports_alternative": [],
                "contested": False, "narrative": ""}

    rows = []
    for name in contributions.columns:
        a = float(contributions.loc[top, name])
        b = float(contributions.loc[runner_up, name])
        if abs(a) < 1e-9 and abs(b) < 1e-9:
            continue
        group = cfg.FEATURE_SPEC[name][0]
        rows.append({
            "feature": name, "group": group, "delta": a - b,
            "leader_z": float(z.loc[top, name]) if name in z.columns else float("nan"),
            "alternative_z": float(z.loc[runner_up, name]) if name in z.columns else float("nan"),
            "leader_value": float(features.loc[top, name]) if name in features.columns else float("nan"),
            "alternative_value": float(features.loc[runner_up, name]) if name in features.columns else float("nan"),
        })
    rows.sort(key=lambda r: -abs(r["delta"]))
    supports_leader = [r for r in rows if r["delta"] > 1e-9][:top_n]
    supports_alt = [r for r in rows if r["delta"] < -1e-9][:top_n]

    gained = sum(r["delta"] for r in rows if r["delta"] > 0)
    lost = -sum(r["delta"] for r in rows if r["delta"] < 0)
    # "Contested" means a material share of the weighted evidence points the
    # other way, not merely that the runner-up is close in total score.
    contested = bool(lost > 0 and gained > 0 and lost / (gained + lost) >= 0.25)

    narrative = ""
    if supports_alt:
        alt_groups = sorted({r["group"] for r in supports_alt})
        lead_groups = sorted({r["group"] for r in supports_leader})
        narrative = (
            f"Competing hypothesis: car {runner_up}. The evidence is split - "
            f"{', '.join(lead_groups)} favour car {top}, while "
            f"{', '.join(alt_groups)} favour car {runner_up} "
            f"({lost / (gained + lost) * 100:.0f}% of the weighted evidence). ")
        if "integrity" in alt_groups:
            narrative += (
                f"Car {runner_up}'s case rests on non-thermodynamic evidence: it reports "
                f"invalid status and sensor dropouts that car {top} does not. That channel is "
                f"deliberately secondary because it is not a measurement of cooling capacity "
                f"and it fires in only half the labelled files, but it is exactly what an "
                f"electrical or sensor fault would look like. If the true defect is electrical "
                f"rather than a loss of charge, car {runner_up} is the better answer. ")
        narrative += (f"Distinguishing test: inspect car {top} for charge loss - sight glass, "
                      f"subcooling, weighed charge - and car {runner_up} for wiring and sensor "
                      f"integrity.")
    return {"alternative": runner_up, "supports_leader": supports_leader,
            "supports_alternative": supports_alt, "contested": contested,
            "evidence_share_against": float(lost / (gained + lost)) if (gained + lost) > 0 else 0.0,
            "narrative": narrative}


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def rank_case(source: str | CasePanel | CleanCase, model: dict | None = None,
              verbose: bool = False) -> dict:
    """
    Localise the refrigerant leak in one case file.

    ``source`` may be a path, an already-loaded :class:`CasePanel` or a
    :class:`CleanCase`, which makes the function cheap to reuse from the
    evaluation loop without re-reading a 34 MB workbook.
    """
    model = load_model() if model is None else model

    if isinstance(source, CleanCase):
        case = source
    else:
        panel = source if isinstance(source, CasePanel) else load_case(source, verbose=verbose)
        case = clean(panel)

    features, ctx = extract(case)
    result = fuse(features,
                  group_weights=model.get("group_weights"),
                  feature_weights=model.get("feature_weights"),
                  w_physics=model.get("w_physics"),
                  w_outlier=model.get("w_outlier"))

    score = result["score"].sort_values(ascending=False)
    diagnosable = list(score.index)
    undiagnosable = [c for c in case.header_cars if c not in diagnosable]
    ranked = diagnosable + undiagnosable

    table = []
    for rank, car in enumerate(ranked, start=1):
        entry = {
            "rank": rank,
            "car": car,
            "score": float(result["score"].get(car, cfg.FUSION["inactive_penalty"])),
            "physics_score": float(result["physics"].get(car, np.nan)),
            "outlier_score": float(result["outlier"].get(car, np.nan)),
            "diagnosable": car in diagnosable,
            "evidence": _explain(car, result["contributions"], features),
        }
        for name in ("elev_mean", "elev_load_stratified", "setpoint_error",
                     "load_sensitivity", "elev_trend", "integrity_loss",
                     "circuit_asym_lift", "suction_excursion"):
            if name in features.columns and car in features.index:
                entry[name] = float(features.loc[car, name])
        table.append(entry)

    top = ranked[0]
    diagnosable_scores = result["score"].reindex(diagnosable).dropna()
    elev_top = float(features.loc[top, "elev_mean"]) if (
        "elev_mean" in features.columns and top in features.index) else float("nan")
    assessment = confidence.assess(diagnosable_scores, evidence_k=elev_top)
    contest = _contest(top, assessment.get("runner_up"), result["contributions"],
                       features, result["z"])

    return {
        "file_id": case.panel.file_id,
        "ranked_cars": "|".join(ranked),
        "ranked_cars_list": ranked,
        "most_likely_faulty_car": top,
        "schema": case.panel.meta.get("schema"),
        "active_groups": ctx["active_groups"],
        "has_refrigerant_evidence": ctx["has_refrigerant"],
        "n_header_cars": len(case.header_cars),
        "n_diagnosable": len(diagnosable),
        "cleaning_report": case.report,
        "features": features,
        "z_scores": result["z"],
        "contributions": result["contributions"],
        "weights": result["weights"],
        "car_table": table,
        "confidence": assessment,
        "competing_hypothesis": contest,
        "verdict": _verdict(top, result["score"], features,
                            _explain(top, result["contributions"], features),
                            ctx["has_refrigerant"], assessment) if top in features.index else
                   f"Car {top} ranked first by fallback ordering: no car had usable telemetry.",
    }


def rank_files(paths: list[str], model: dict | None = None,
               verbose: bool = False) -> list[dict]:
    model = load_model() if model is None else model
    return [rank_case(path, model=model, verbose=verbose) for path in paths]
