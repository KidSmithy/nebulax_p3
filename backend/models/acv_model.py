"""
backend/models/acv_model.py
===============================================================================
Air Conditioning & Ventilation (ACV) Subsystem Model.

Two distinct capabilities live here:

1. evaluate_telemetry(...)
   The live 10 Hz digital-twin path. One rooftop ACV pack, evaluated for
   thermodynamic heat exchange, COP efficiency and filter obstruction.
   Called by InferenceBroker on every frame. Unchanged behaviour.

2. localise_leak(...) / rank_consist_window(...)
   Refrigerant-leakage localisation across a consist. This is now a thin
   adapter over the ACV2 package (``ACV2/acv2/``), which is the single
   reference implementation.

-------------------------------------------------------------------------------
WHY THIS FILE NO LONGER CONTAINS ANY PHYSICS
-------------------------------------------------------------------------------
It used to carry a standalone copy of the five-feature scoring core from
ACV/acv_physics_ranker.py, with a parity test to stop the two copies drifting.
That was a workable arrangement for two copies and a bad one for three. The
physics now lives in exactly one place and this module translates its output
into the backend's response contract.

The upgrade is not cosmetic. ACV2 scores 1.0000 on the six labelled cases under
leave-one-file-out cross validation against a 0.5625 random baseline, using 20
channels across six evidence groups rather than five thermal statistics, and it
reads refrigerant circuit pressures on the rich-schema files where the previous
core discarded them.

-------------------------------------------------------------------------------
THE BUG THIS PORT FIXES
-------------------------------------------------------------------------------
The previous version derived its confidence, and therefore its operational
verdict, from the raw top-1 margin:

    margin >= 2.0 -> "CLEAR"      margin >= 0.5 -> "MODERATE"
    and  predict_from_csv:  confidence in ("CLEAR", "MODERATE") -> ACTION_NEEDED

ACV2's audit measured that statistic directly. Deleting the known faulty car
from each labelled file and re-ranking the healthy siblings produces a margin of
**the same size** - median ratio 1.00x, and on one case the healthy-only consist
separates ten times more strongly than the genuine fault. The cause is
structural: a peer-consensus ranker returns whichever car is most anomalous
relative to its siblings, and exactly one car is always the warmest, so a winner
with a margin appears whether or not anything is broken.

So the margin measures spread, not fault presence, and this backend was capable
of telling an operator "ACTION_NEEDED, service Car 04, CLEAR confidence" about a
consist in perfect health.

Confidence is now the null-referenced statistic from ``acv2.confidence``: a
scale-free Dixon-Q separation compared against 41 consists known to contain no
fault, cross-checked against the measured detection limit. A weak verdict is
reported as WATCH with the leading suspect still named - not silently promoted to
ACTION_NEEDED, and not demoted to "all normal" either, because a weakly
separated ranking is still the best available ordering of the evidence.

-------------------------------------------------------------------------------
NO MODEL ARTEFACT IS REQUIRED
-------------------------------------------------------------------------------
There is nothing to unpickle and no training step at inference time:

  * the ranker is closed-form - feature signs come from the vapour-compression
    energy balance and the weights are a fixed physical prior;
  * ``ACV2/artifacts/acv2_model.joblib`` holds weights only, and because
    calibration returns the prior unchanged, its absence changes no number;
  * the two confidence artefacts (``null_calibration.json``,
    ``detection_limit.json``) are optional. Without them confidence is reported
    as ``uncalibrated`` and no probability is attached, rather than a number
    being invented.

The only hard dependency is pandas plus scikit-learn, both already required by
the backend.
===============================================================================
"""

from __future__ import annotations

import os
import sys

import numpy as np

from backend.models.submission import make_submission, merge_submissions

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:                                      # pragma: no cover
    _PANDAS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Bridge to the ACV2 reference implementation
# ---------------------------------------------------------------------------
# ACV2 is a self-contained package at the repository root rather than a module
# inside `backend`, because it is also the standalone hackathon deliverable with
# its own CLI, tests and reports. Adding its directory to sys.path keeps one
# copy of the physics without turning the backend into its parent.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_CANDIDATE_ACV2_DIRS = [
    os.path.join(_REPO_ROOT, "ACV2"),
    os.path.join(_BACKEND_DIR, "ACV2"),
    "/app/ACV2",
    "/app/backend/ACV2",
]

_ACV2_DIR = next((d for d in _CANDIDATE_ACV2_DIRS if os.path.isdir(d)), os.path.join(_REPO_ROOT, "ACV2"))

for _d in _CANDIDATE_ACV2_DIRS:
    if os.path.isdir(_d) and _d not in sys.path:
        sys.path.insert(0, _d)

try:
    from acv2 import config as acv2_cfg
    from acv2 import confidence as acv2_confidence
    from acv2.ranker import load_model as _acv2_load_model
    from acv2.ranker import rank_case as _acv2_rank_case
    _ACV2_AVAILABLE = True
    _ACV2_IMPORT_ERROR = None
except Exception as exc:                                 # pragma: no cover
    _ACV2_AVAILABLE = False
    _ACV2_IMPORT_ERROR = exc
    acv2_cfg = None
    acv2_confidence = None


# ---------------------------------------------------------------------------
# Live-twin thresholds (10 Hz path only - unrelated to leak localisation)
# ---------------------------------------------------------------------------
#: Nominal supply/return temperature split of a healthy pack, deg C.
NOMINAL_DELTA_C = 8.5
#: Nominal compressor draw at that split, kW.
NOMINAL_COMPRESSOR_KW = 4.5


# ---------------------------------------------------------------------------
# Confidence vocabulary
# ---------------------------------------------------------------------------
# ACV2 reports a calibrated level; the HUD and the shared upload contract speak
# the older four-word vocabulary. This is the only mapping between them, so the
# words the operator reads can never disagree with the statistic behind them.
#
# Deliberately *not* mapped from the margin. That is the bug this port fixes.
CONFIDENCE_FROM_LEVEL = {
    "strong": "CLEAR",
    "moderate": "MODERATE",
    "weak": "AMBIGUOUS",
    "uncalibrated": "UNCALIBRATED",
}

#: Levels at which a named car is worth dispatching a technician for.
ACTIONABLE_LEVELS = ("strong", "moderate")

#: Plain-language copy for the HUD / AI insight layer, per ACV2 channel.
FEATURE_LABELS = {
    "elev_mean": "Warmer than the other cars",
    "elev_load_stratified": "Warmer than the other cars, at matched workload",
    "elev_p90": "Worst-case warm excursion",
    "elev_persistence": "How long it stays warm",
    "setpoint_error": "Missing its own target temperature",
    "load_sensitivity": "Falls further behind as the day heats up",
    "capacity_shortfall": "Share of the demanded cooling it fails to deliver",
    "pulldown_rate": "How fast it can cool the cabin down",
    "greybox_cool_rate": "Identified cooling power",
    "elev_trend": "Getting worse over the record",
    "cusum_fraction": "How long ago the drift started",
    "full_demand_frac": "How often it is driven at full cooling",
    "circuit_asym_lift": "Its two refrigeration circuits disagree",
    "circuit_asym_high": "Condensing pressure mismatch between circuits",
    "circuit_asym_ratio": "Compression ratio mismatch between circuits",
    "suction_excursion": "Suction pressure keeps collapsing",
    "lift_deficit": "Weakest circuit underperforms the other cars",
    "compressor_duty": "Compressor running harder than its peers",
    "compressor_cycling": "Compressor short-cycling",
    "integrity_loss": "Reporting invalid readings and dropouts",
}


def _require_acv2():
    if not _PANDAS_AVAILABLE:
        raise RuntimeError("pandas is required for ACV leak localisation")
    if not _ACV2_AVAILABLE:
        raise RuntimeError(
            f"ACV2 package not importable from {_ACV2_DIR!r}: {_ACV2_IMPORT_ERROR}. "
            f"Leak localisation is unavailable; the live 10 Hz path is unaffected."
        )


def _f(value, digits: int = 3):
    """Round to JSON, mapping NaN/inf to None so the response stays valid JSON."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return round(out, digits) if np.isfinite(out) else None


# ---------------------------------------------------------------------------
# Subsystem model
# ---------------------------------------------------------------------------
class ACVSubsystemModel:
    """
    ACV subsystem model.

    Live twin path  : evaluate_telemetry()
    Diagnosis path  : localise_leak() / rank_consist_window()
    """

    #: Reported to the API so the UI can state which method is in use.
    METHOD = "acv2_peer_consensus_physics_ranker"
    METHOD_VERSION = "3.0.0"

    def __init__(self):
        self.leak_model_available = _PANDAS_AVAILABLE and _ACV2_AVAILABLE
        self._model = _acv2_load_model() if self.leak_model_available else None
        self._null = (acv2_confidence.load_null_calibration()
                      if self.leak_model_available else None)
        self._limit = (acv2_confidence.load_detection_limit()
                       if self.leak_model_available else None)

    # -- capability description -------------------------------------------
    def describe_method(self) -> dict:
        """Metadata for /api/health and the AI insight layer."""
        groups, weights, thresholds = {}, {}, {}
        if self.leak_model_available:
            weights = dict(self._model.get("feature_weights") or {})
            groups = {
                group: [name for name, spec in acv2_cfg.FEATURE_SPEC.items()
                        if spec[0] == group]
                for group in acv2_cfg.FEATURE_GROUPS
            }
            thresholds = {
                "elevation_threshold_c": acv2_cfg.PHYSICS["elevation_threshold_c"],
                "persist_window_min": acv2_cfg.PHYSICS["persist_window_min"],
                "t_in_min_c": acv2_cfg.CLEANING["t_in_min"],
                "t_in_max_c": acv2_cfg.CLEANING["t_in_max"],
                "cusum_k_c": acv2_cfg.PHYSICS["cusum_k_c"],
                "cusum_h_c": acv2_cfg.PHYSICS["cusum_h_c"],
            }

        return {
            "method": self.METHOD,
            "version": self.METHOD_VERSION,
            "implementation": "ACV2/acv2",
            "fitted_parameters": 0,
            "requires_training_data": False,
            "model_artefact": None,
            "leak_localisation_available": self.leak_model_available,
            "evidence_groups": groups,
            "physics_weights": weights,
            "zero_weight_channels": (
                [name for name, spec in acv2_cfg.FEATURE_SPEC.items() if spec[2] <= 0]
                if self.leak_model_available else []
            ),
            "thresholds": thresholds,
            "feature_labels": dict(FEATURE_LABELS),
            "confidence": {
                "statistic": "dixon_q_vs_fault_free_null",
                "null_calibrated": self._null is not None,
                "n_null_consists": len((self._null or {}).get("dixon_q_null", [])),
                "detection_limit_available": self._limit is not None,
                "note": (
                    "Confidence is referenced to consists known to contain no fault. It is "
                    "NOT the top-1 margin: healthy consists were measured to produce margins "
                    "of the same size, so the margin carries no information about whether a "
                    "fault is present at all."
                ),
            },
            "summary": (
                "Ranks every car in a consist by a weighted robust-z pool of up to 20 "
                "physics channels - cabin thermal deficit, cooling capacity, progression, "
                "refrigerant circuit pressures and data integrity - each measured against "
                "its sibling cars under matched conditions. No fitted parameters, no model "
                "artefact."
            ),
        }

    # -- live 10 Hz twin path ---------------------------------------------
    # Unchanged. This path evaluates a single rooftop pack from the simulator's
    # own supply/return/compressor channels and has no consist to compare
    # against, so none of the peer-consensus physics applies to it.
    def evaluate_telemetry(self, supply_temp: float, return_temp: float,
                           compressor_kw: float, degraded: bool = False) -> dict:
        delta_temp = round(float(return_temp - supply_temp), 1)

        efficiency_ratio = delta_temp / max(NOMINAL_DELTA_C, 1.0)
        power_ratio = compressor_kw / NOMINAL_COMPRESSOR_KW

        if degraded:
            efficiency_rating = round(float(np.clip(efficiency_ratio * 0.72, 0.40, 0.75)), 2)
            anomaly_score = round(float(np.clip(
                (1.0 - efficiency_rating) * 1.3 + (power_ratio - 1.0) * 0.3, 0.55, 0.95)), 2)
            fault_type = "REFRIGERANT_LEAKAGE" if delta_temp < 6.0 else "FILTER_CLOGGING"
        else:
            efficiency_rating = round(float(np.clip(efficiency_ratio * 0.95, 0.75, 0.99)), 2)
            anomaly_score = round(float(np.clip(1.0 - efficiency_rating, 0.05, 0.35)), 2)
            fault_type = "NONE"

        return {
            "unit_id": "ACV_PACK_1",
            "supply_temp_c": round(float(supply_temp), 1),
            "return_temp_c": round(float(return_temp), 1),
            "delta_temp_c": delta_temp,
            "compressor_power_kw": round(float(compressor_kw), 2),
            "efficiency_rating": efficiency_rating,
            "anomaly_score": anomaly_score,
            "fault_type": fault_type,
        }

    # -- consist leak localisation ----------------------------------------
    def localise_leak(self, source, file_id: str = None) -> dict:
        """
        Rank every car in an ACV consist from most to least likely to be
        leaking refrigerant.

        source:  a path to an .xlsx/.csv consist file in the PS3 ACV format,
                 OR an already-loaded pandas DataFrame in that same format
                 (used by the /api upload route, which parses CSV bytes first).
        file_id: label echoed back in the result; defaults to the file's base
                 name, or 'uploaded_data' for a DataFrame.

        Returns ranked_cars (pipe separated, as the submission format wants),
        the leading suspect, a calibrated confidence, the competing hypothesis
        where the evidence is split, and a per-car diagnostic table suitable for
        direct display in the HUD.
        """
        _require_acv2()

        if isinstance(source, pd.DataFrame):
            label = file_id or "uploaded_data"
            if source.empty:
                raise ValueError(f"No usable cabin temperature telemetry in {label}")
            payload = source
        else:
            if not os.path.exists(source):
                raise FileNotFoundError(source)
            label = file_id or os.path.basename(source)
            payload = source

        try:
            result = _acv2_rank_case(payload, model=self._model, file_id=label)
        except Exception as exc:
            # A schema that carries no recognisable ACV columns must fail loudly
            # rather than rank noise, and the message must name the expected
            # header form so an operator can fix their export.
            raise ValueError(
                f"{label}: could not localise a leak ({exc}). Expected headers of the "
                f"form 'Car <NN> - Indoor Average Temperature'."
            ) from exc

        if not result.get("ranked_cars_list"):
            raise ValueError(f"No usable cabin temperature telemetry in {label}")
        if result.get("n_diagnosable", 0) < 1:
            raise ValueError(f"No usable cabin temperature telemetry in {label}")

        return self._assemble(result, file_id=label)

    def rank_consist_window(self, cabin_temps: dict, setpoints: dict = None,
                            cooling_flags: dict = None, outdoor_temps: dict = None,
                            sample_interval_s: float = 30.0) -> dict:
        """
        Same physics, driven by a live rolling buffer instead of a file.

        cabin_temps:   {car_id: [T0, T1, ...]} equal-length cabin temperature
                       series, oldest first
        setpoints:     {car_id: [...]} or {car_id: scalar}; omitted -> the
                       setpoint and capacity terms contribute nothing
        cooling_flags: {car_id: [bool, ...]} or {car_id: bool}; omitted ->
                       every sample is treated as active cooling
        outdoor_temps: {car_id: [...]} or scalar per car; omitted -> the
                       load-stratified and load-sensitivity channels cannot be
                       computed and are reported as unavailable rather than
                       guessed
        sample_interval_s: seconds between samples, used to size the rolling
                       persistence window and the CUSUM slots correctly

        Implemented by assembling the buffer into the dataset's own
        ``Car <NN> - <parameter>`` header layout and handing it to the identical
        code path as a file, so the live twin and the offline diagnosis can never
        disagree about the same numbers.

        Intended for a consist-aware live twin. The current single-pack twin does
        not emit per-car cabin temperatures, so nothing in the 10 Hz path calls
        this yet.
        """
        _require_acv2()
        if not cabin_temps:
            raise ValueError("cabin_temps is empty")

        cars = list(cabin_temps)
        length = max(len(list(v)) for v in cabin_temps.values())
        interval = float(sample_interval_s) if sample_interval_s else 30.0

        frame = {"Time": pd.date_range("2000-01-01", periods=length,
                                       freq=pd.Timedelta(seconds=interval))}

        def column(src, car, default):
            if src is None:
                return default
            value = src.get(car, default)
            if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
                return pd.Series(list(value)).reindex(range(length)).to_numpy()
            return value

        for car in cars:
            frame[f"Car {car} - Indoor Average Temperature"] = (
                pd.Series(list(cabin_temps[car])).reindex(range(length)).to_numpy())

            setpoint = column(setpoints, car, np.nan)
            frame[f"Car {car} - ACV Control Temperature (Cooling)"] = setpoint

            outdoor = column(outdoor_temps, car, np.nan)
            frame[f"Car {car} - Outdoor Average Temperature"] = outdoor

            flags = column(cooling_flags, car, True)
            if isinstance(flags, np.ndarray):
                mode = np.where(flags.astype(bool), "Automatic Cooling", "Ventilation")
            else:
                mode = "Automatic Cooling" if bool(flags) else "Ventilation"
            frame[f"Car {car} - ACV Running Mode"] = mode
            frame[f"Car {car} - ACV Information Valid"] = "Valid"

        result = _acv2_rank_case(pd.DataFrame(frame), model=self._model,
                                file_id="live_window")
        return self._assemble(result, file_id="live_window")

    # -- shared result assembly -------------------------------------------
    def _assemble(self, result: dict, file_id: str) -> dict:
        """Translate an ACV2 result into the backend's response contract."""
        ranked = list(result["ranked_cars_list"])
        features = result["features"]
        conf = result.get("confidence") or {}
        contest = result.get("competing_hypothesis") or {}
        level = conf.get("level", "uncalibrated")

        active = [entry["car"] for entry in result["car_table"] if entry["diagnosable"]]
        unassessable = [c for c in ranked if c not in active]

        def feature(car, name):
            if name in features.columns and car in features.index:
                return features.loc[car, name]
            return float("nan")

        diagnostics = []
        for entry in result["car_table"]:
            car = entry["car"]
            if not entry["diagnosable"]:
                diagnostics.append({
                    "rank": entry["rank"], "car": car,
                    "health_index": None, "assessable": False,
                })
                continue
            diagnostics.append({
                "rank": entry["rank"],
                "car": car,
                "health_index": _f(entry["score"]),
                # Legacy keys, kept so existing HUD code and saved findings
                # keep rendering. Each is the ACV2 channel that measures the
                # same physical quantity the old five-feature core measured.
                "mean_rel_c": _f(feature(car, "elev_mean")),
                "mean_t_minus_set_c": _f(feature(car, "setpoint_error")),
                "p95_rel_c": _f(feature(car, "elev_p90")),
                "persistence_pct": _f(100.0 * feature(car, "elev_persistence"), 1),
                # New ACV2 evidence.
                "elev_load_stratified_c": _f(feature(car, "elev_load_stratified")),
                "capacity_shortfall": _f(feature(car, "capacity_shortfall")),
                "load_sensitivity": _f(feature(car, "load_sensitivity")),
                "elev_trend_c_per_day": _f(feature(car, "elev_trend")),
                "cusum_fraction": _f(feature(car, "cusum_fraction")),
                "integrity_loss": _f(feature(car, "integrity_loss")),
                "circuit_asym_lift": _f(feature(car, "circuit_asym_lift")),
                "suction_excursion": _f(feature(car, "suction_excursion")),
                "physics_score": _f(entry.get("physics_score")),
                "evidence": [
                    {
                        "feature": e["feature"],
                        "label": FEATURE_LABELS.get(e["feature"], e["feature"]),
                        "group": e["group"],
                        "value": _f(e["value"]),
                        "contribution": _f(e["contribution"]),
                    }
                    for e in entry.get("evidence", [])
                ],
                "assessable": True,
            })

        top = ranked[0] if ranked else None
        summary = result.get("verdict") or "No car reported usable cabin telemetry."

        detection = conf.get("detection") or {}
        return {
            "file_id": file_id,
            "method": self.METHOD,
            "method_version": self.METHOD_VERSION,
            "schema": result.get("schema"),
            "ranked_cars": "|".join(ranked),
            "ranked_cars_list": ranked,
            "most_likely_faulty_car": top,

            # Confidence. `confidence` keeps the legacy vocabulary for the HUD;
            # everything below it is the measurement that vocabulary came from.
            "confidence": CONFIDENCE_FROM_LEVEL.get(level, "UNCALIBRATED"),
            "confidence_level": level,
            "confidence_statement": acv2_confidence.describe(conf) if conf else "",
            "separation_q": _f(conf.get("dixon_q")),
            "null_p_value": _f(conf.get("null_p_value")),
            "n_null_consists": conf.get("n_null_samples", 0),
            "null_calibrated": bool(conf.get("n_null_samples")),
            # Retained for backward compatibility and for display, explicitly
            # flagged as not being the confidence signal.
            "confidence_margin": _f(conf.get("margin")),
            "margin_is_not_confidence": True,
            "detection_limit": {
                "evidence_k": _f(detection.get("evidence_k")),
                "expected_top1_rate": _f(detection.get("expected_top1_rate")),
                "reliable_from_k": _f(detection.get("reliable_from_k"), 2),
                "below_detection_limit": detection.get("below_detection_limit"),
            },

            "competing_hypothesis": {
                "alternative": contest.get("alternative"),
                "contested": bool(contest.get("contested")),
                "evidence_share_against": _f(contest.get("evidence_share_against")),
                "narrative": contest.get("narrative", ""),
                "supports_alternative": [
                    {"feature": r["feature"],
                     "label": FEATURE_LABELS.get(r["feature"], r["feature"]),
                     "group": r["group"],
                     "alternative_z": _f(r["alternative_z"], 2),
                     "leader_z": _f(r["leader_z"], 2)}
                    for r in contest.get("supports_alternative", [])
                ],
            },

            "evidence_groups_used": result.get("active_groups", []),
            "has_refrigerant_evidence": result.get("has_refrigerant_evidence", False),
            "consist_size": len(ranked),
            "active_cars": active,
            "unassessable_cars": unassessable,
            "car_diagnostics": diagnostics,
            "diagnostic_summary": summary,
            "sample_interval_s": (result.get("cleaning_report") or {}).get("dt_seconds"),
            "n_rows_used": (result.get("cleaning_report") or {}).get("rows_kept"),
        }

    # -- upload paths ------------------------------------------------------
    def predict_from_csv(self, file_source, file_name: str = "acv_data.csv") -> dict:
        """
        Runs ACV refrigerant-leak localisation from an uploaded CSV or Excel
        (.xlsx) case file.

        ACV2's loader picks its parser by file extension, so raw upload bytes
        are spooled to a temp file with the ORIGINAL extension preserved rather
        than forced through pd.read_csv - that would corrupt/reject a binary
        .xlsx workbook, which is the format the real ACV test cases ship in.
        """
        import tempfile

        suffix = os.path.splitext(file_name)[1] or ".xlsx"
        if isinstance(file_source, (bytes, bytearray)):
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(file_source)
                tmp_path = tmp.name
            try:
                leak = self.localise_leak(tmp_path, file_id=file_name)
            finally:
                os.unlink(tmp_path)
        elif isinstance(file_source, str) and not os.path.exists(file_source):
            # A CSV passed as text rather than a path or bytes.
            import io
            leak = self.localise_leak(pd.read_csv(io.StringIO(file_source)),
                                      file_id=file_name)
        else:
            leak = self.localise_leak(file_source, file_id=file_name)

        top_car = leak.get("most_likely_faulty_car")
        level = leak.get("confidence_level", "uncalibrated")
        confidence = leak.get("confidence", "UNCALIBRATED")
        q = leak.get("separation_q")
        p = leak.get("null_p_value")
        below_limit = (leak.get("detection_limit") or {}).get("below_detection_limit")

        # Verdict banding.
        #
        # The old logic promoted anything with margin >= 0.5 to ACTION_NEEDED and
        # everything else to "all cars in thermal equilibrium". Both halves were
        # wrong: the first because the margin does not indicate a fault, and the
        # second because a weakly separated ranking is not evidence of health.
        # A weak verdict now lands on WATCH with the suspect still named.
        if top_car and level in ACTIONABLE_LEVELS:
            status = "ACTION_NEEDED"
            action = "ACTION_REPLACE_FILTER"
            anomaly_score = 0.78
            summary = (
                f"Consist refrigerant evaluation for '{file_name}': Car {top_car} is the "
                f"most likely leak location, at {confidence} confidence "
                f"(separation Q={q}, matched or exceeded by "
                f"{'' if p is None else f'{p * 100:.0f}% of '}known-healthy consists). "
                f"Recommend servicing the AC pack on Car {top_car}."
            )
        elif top_car:
            status = "WATCH"
            action = "ACTION_INSPECT_ACV_PACK"
            anomaly_score = 0.42
            summary = (
                f"Consist refrigerant evaluation for '{file_name}': Car {top_car} is the "
                f"leading suspect, but the evidence is inconclusive. Its separation from "
                f"the rest of the consist (Q={q}) is matched by "
                f"{'many' if p is None else f'{p * 100:.0f}% of'} consists known to contain "
                f"no fault, so this ordering is the best reading of thin evidence rather "
                f"than a confirmed diagnosis. "
                + ("The underlying thermal deficit is below the magnitude at which recovery "
                   "is reliable, so treat it as a watch item. " if below_limit else "")
                + f"Suggest a routine inspection of Car {top_car} rather than a callout."
            )
        else:
            status = "GOOD"
            action = "NONE"
            anomaly_score = 0.15
            summary = (
                f"Consist refrigerant evaluation for '{file_name}': no car could be "
                f"assessed from this file."
            )

        contest = leak.get("competing_hypothesis") or {}
        if contest.get("contested") and contest.get("narrative"):
            summary += " " + contest["narrative"]

        return {
            "subsystem": "acv",
            "file_name": file_name,
            "status": status,
            "verdict": status,
            "anomaly_score": anomaly_score,
            "most_likely_faulty_car": top_car,
            "confidence": confidence,
            "confidence_level": level,
            "confidence_margin": leak.get("confidence_margin"),
            "separation_q": q,
            "null_p_value": p,
            "n_null_consists": leak.get("n_null_consists", 0),
            "detection_limit": leak.get("detection_limit"),
            "competing_hypothesis": contest,
            "evidence_groups_used": leak.get("evidence_groups_used", []),
            "has_refrigerant_evidence": leak.get("has_refrigerant_evidence", False),
            "recommended_action": action,
            "conductor_summary": summary,
            "ranked_cars": leak.get("ranked_cars_list", []),
            "car_diagnostics": leak.get("car_diagnostics", []),
            "submission": make_submission(
                "acv_predictions.csv",
                ["file_id", "ranked_cars"],
                [[file_name, "|".join(leak.get("ranked_cars_list", []))]],
            ),
        }

    def predict_batch(self, file_list: list, batch_name: str = "acv_batch.zip") -> dict:
        """
        Evaluates a batch of ACV consist run CSV/Excel files (e.g. from an
        uploaded ZIP).

        file_list: list of (file_name, file_bytes)
        """
        results, failures = [], []
        for fname, fbytes in file_list:
            try:
                results.append(self.predict_from_csv(fbytes, fname))
            except Exception as exc:
                failures.append({"file": fname, "error": str(exc)})
                print(f"[ACVSubsystemModel] error processing {fname} in batch: {exc}")

        if not results:
            raise ValueError(f"Could not parse any valid ACV data files from '{batch_name}'.")

        total_files = len(results)
        flagged = [r for r in results if r["status"] != "GOOD"]
        action_needed = [r for r in results if r["status"] == "ACTION_NEEDED"]

        # Worst file is chosen by calibrated evidence, not by margin: a bigger
        # margin does not mean a more likely fault. Order is
        # (actionable, then how rare the separation is among healthy consists).
        def severity(r):
            level_rank = {"strong": 3, "moderate": 2, "weak": 1}.get(
                r.get("confidence_level"), 0)
            p = r.get("null_p_value")
            rarity = (1.0 - p) if isinstance(p, (int, float)) else 0.0
            return (level_rank, rarity, r.get("separation_q") or 0.0)

        worst_item = max(results, key=severity)
        worst_file = worst_item["file_name"]

        if action_needed:
            verdict = "ACTION_NEEDED"
            action = "ACTION_REPLACE_FILTER"
            conductor_summary = (
                f"Batch evaluation of {total_files} ACV trip log(s) in '{batch_name}': "
                f"{len(action_needed)} log(s) carry a refrigerant-leak diagnosis at usable "
                f"confidence. Primary suspect is Car "
                f"{worst_item.get('most_likely_faulty_car')} in '{worst_file}' "
                f"({worst_item.get('confidence')}, separation Q="
                f"{worst_item.get('separation_q')}). Recommend servicing the AC pack on "
                f"Car {worst_item.get('most_likely_faulty_car')}."
            )
        elif flagged:
            verdict = "WATCH"
            action = "ACTION_INSPECT_ACV_PACK"
            conductor_summary = (
                f"Batch evaluation of {total_files} ACV trip log(s) in '{batch_name}': "
                f"every log produces a leading suspect, but none separates from its consist "
                f"more cleanly than fault-free consists routinely do. Nothing here justifies "
                f"a callout. Highest-ranking watch item is Car "
                f"{worst_item.get('most_likely_faulty_car')} in '{worst_file}'."
            )
        else:
            verdict = "GOOD"
            action = "NONE"
            conductor_summary = (
                f"Batch evaluation of {total_files} ACV trip log(s) in '{batch_name}': "
                f"no car could be assessed in any log."
            )

        if failures:
            conductor_summary += (
                f" {len(failures)} file(s) could not be parsed and were skipped.")

        batch_breakdown = [
            {
                "file": r["file_name"],
                "faulty_car": r.get("most_likely_faulty_car"),
                "confidence": r.get("confidence"),
                "confidence_level": r.get("confidence_level"),
                "separation_q": r.get("separation_q"),
                "null_p_value": r.get("null_p_value"),
                "margin": r.get("confidence_margin"),
                "verdict": r["verdict"],
            }
            for r in results
        ]

        return {
            "subsystem": "acv",
            "file_name": batch_name,
            "is_batch": True,
            "status": verdict,
            "verdict": verdict,
            "total_files": total_files,
            "anomaly_count": len(flagged),
            "action_needed_count": len(action_needed),
            "most_likely_faulty_car": worst_item.get("most_likely_faulty_car"),
            "confidence": worst_item.get("confidence"),
            "confidence_level": worst_item.get("confidence_level"),
            "confidence_margin": worst_item.get("confidence_margin"),
            "separation_q": worst_item.get("separation_q"),
            "null_p_value": worst_item.get("null_p_value"),
            "worst_file": worst_file,
            "anomaly_score": worst_item.get("anomaly_score", 0.15),
            "recommended_action": action,
            "conductor_summary": conductor_summary,
            "batch_items": batch_breakdown,
            "submission": merge_submissions(results),
            "skipped_files": failures,
        }
