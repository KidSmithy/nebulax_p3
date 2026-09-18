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
            model = dict(DEFAULT_MODEL)
            model.update(joblib.load(path))
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
             evidence: list[dict], has_refrigerant: bool) -> str:
    ordered = score.sort_values(ascending=False)
    margin = float(ordered.iloc[0] - ordered.iloc[1]) if len(ordered) > 1 else float("nan")
    elev = features.loc[top, "elev_mean"] if "elev_mean" in features.columns else np.nan
    bits = [f"Car {top} is the most likely refrigerant-leak location "
            f"(score {ordered.iloc[0]:.3f}, margin {margin:.3f} over car {ordered.index[1]})."
            if len(ordered) > 1 else f"Car {top} is the most likely refrigerant-leak location."]
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
    return " ".join(bits)


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
        "verdict": _verdict(top, result["score"], features,
                            _explain(top, result["contributions"], features),
                            ctx["has_refrigerant"]) if top in features.index else
                   f"Car {top} ranked first by fallback ordering: no car had usable telemetry.",
    }


def rank_files(paths: list[str], model: dict | None = None,
               verbose: bool = False) -> list[dict]:
    model = load_model() if model is None else model
    return [rank_case(path, model=model, verbose=verbose) for path in paths]
