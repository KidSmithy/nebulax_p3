"""
backend/models/test_acv_leak_model.py
===============================================================================
Parity and behaviour tests for the ACV subsystem model.

backend/models/acv_model.py no longer carries its own copy of the physics: it is
an adapter over the ACV2 package. So the parity target has moved. This suite now
asserts that the backend adapter reproduces ACV2's own ranking and confidence
exactly, which is the property that matters - if the adapter silently diverged,
the HUD would disagree with the reference implementation and with the reports in
ACV2/reports/.

It also checks the things that would quietly break the subsystem:
  - the live 10 Hz twin contract (evaluate_telemetry) still holds
  - the physics points the right way (a deliberately hot car must rank first)
  - the live-buffer path agrees with the file path on the same data
  - confidence is NOT derived from the top-1 margin, which was measured to carry
    no information about whether a fault is present at all
  - a weakly separated verdict is reported as WATCH with the suspect named,
    rather than promoted to ACTION_NEEDED or demoted to "all normal"

Run:  python -m pytest backend/models/test_acv_leak_model.py -q
      (or, with no pytest installed:  python backend/models/test_acv_leak_model.py)
===============================================================================
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "PS3", "02_Datasets", "ACV")
TEST_CASE = os.path.join(DATA, "Test", "acv_test_case.xlsx")

sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ACV2"))

from acv_model import (  # noqa: E402
    ACTIONABLE_LEVELS,
    ACVSubsystemModel,
    CONFIDENCE_FROM_LEVEL,
    FEATURE_LABELS,
)

from acv2 import confidence as acv2_confidence  # noqa: E402
from acv2 import config as acv2_cfg             # noqa: E402
from acv2.ranker import rank_case               # noqa: E402


# ---------------------------------------------------------------------------
# parity with the ACV2 reference implementation
# ---------------------------------------------------------------------------
def test_backend_ranking_matches_acv2_exactly():
    """
    The adapter must not change a single number. Same ranking, same separation,
    same p-value as ACV2 produces on its own.

    acv_case_04 is excluded only because it is 33 MB and slow to parse; the files
    used here cover both schema variants that matter at inference.
    """
    model = ACVSubsystemModel()
    assert model.leak_model_available, "ACV2 must be importable from the backend"

    cases = [TEST_CASE]
    for name in ("acv_case_01.xlsx", "acv_case_05.xlsx"):
        p = os.path.join(DATA, "Train", name)
        if os.path.exists(p):
            cases.append(p)
    assert cases, "no ACV case files found to test against"

    for path in cases:
        mine = model.localise_leak(path)
        theirs = rank_case(path)

        assert mine["ranked_cars"] == theirs["ranked_cars"], (
            f"{os.path.basename(path)}: ranking differs - "
            f"backend {mine['ranked_cars']} vs ACV2 {theirs['ranked_cars']}"
        )
        assert mine["most_likely_faulty_car"] == theirs["most_likely_faulty_car"]
        assert mine["confidence_level"] == theirs["confidence"]["level"]
        assert abs(mine["separation_q"] - theirs["confidence"]["dixon_q"]) < 1e-3
        assert mine["consist_size"] == len(theirs["ranked_cars_list"])

        # per-car feature values must survive the translation
        feats = theirs["features"]
        for row in mine["car_diagnostics"]:
            if not row["assessable"]:
                continue
            c = row["car"]
            assert abs(row["mean_rel_c"] - round(float(feats.loc[c, "elev_mean"]), 3)) < 1e-9
            assert abs(row["mean_t_minus_set_c"]
                       - round(float(feats.loc[c, "setpoint_error"]), 3)) < 1e-9


def test_backend_uses_acv2_not_a_local_copy_of_the_physics():
    """
    Guard against the duplication returning. The adapter must not define its own
    feature set, weights or scoring core.
    """
    import acv_model

    for banned in ("PHYSICS_WEIGHTS", "FEATURE_ORDER", "features_from_frames",
                   "score_features", "robust_z", "clean_indoor_temperatures"):
        assert not hasattr(acv_model, banned), (
            f"acv_model.{banned} reintroduces a local copy of the physics; "
            f"it belongs in ACV2/acv2/ only"
        )
    assert acv_model._ACV2_AVAILABLE is True


def test_feature_labels_cover_every_acv2_channel():
    """Every channel the HUD can surface needs plain-language copy."""
    missing = [name for name in acv2_cfg.FEATURE_SPEC if name not in FEATURE_LABELS]
    assert not missing, f"no plain-language label for: {missing}"


# ---------------------------------------------------------------------------
# ranking contract
# ---------------------------------------------------------------------------
def test_ranking_is_a_full_permutation():
    """Every car in the file's headers must be ranked, or it scores zero."""
    model = ACVSubsystemModel()
    res = model.localise_leak(TEST_CASE)
    cars = res["ranked_cars"].split("|")
    assert len(cars) == 8
    assert len(set(cars)) == 8, "duplicate car id in ranking"
    assert all(len(c) == 2 and c.isdigit() for c in cars), "car ids must be two-digit"


def test_physics_points_the_right_way():
    """
    Direction check: the whole method rests on 'a leaking car runs WARM'. A
    deliberately warm car must rank first, and a deliberately cold one last.
    """
    model = ACVSubsystemModel()
    rng = np.random.default_rng(7)
    n = 400
    temps = {f"{i:02d}": 24.0 + rng.normal(0, 0.05, n) for i in range(1, 9)}
    temps["04"] = temps["04"] + 1.2          # the planted leak
    temps["07"] = temps["07"] - 0.8          # over-cooling, must NOT be blamed

    res = model.rank_consist_window(
        cabin_temps={k: list(v) for k, v in temps.items()},
        setpoints={k: 24.0 for k in temps},
        sample_interval_s=30.0,
    )
    order = res["ranked_cars"].split("|")
    assert order[0] == "04", f"warm car not ranked first, got {res['ranked_cars']}"
    assert order[-1] == "07", f"coldest car not ranked last, got {res['ranked_cars']}"


def test_live_window_agrees_with_the_file_path():
    """
    The live-buffer path assembles the dataset's own header layout and goes
    through the identical code path, so feeding a file's cabin temperatures back
    through it must reproduce that file's ranking.
    """
    model = ACVSubsystemModel()
    from acv2.cleaning import clean
    from acv2.io_loader import load_case

    case = clean(load_case(TEST_CASE))
    active = case.cars
    live = model.rank_consist_window(
        cabin_temps={c: case.t_in[c].tolist() for c in active},
        setpoints={c: case.t_set[c].tolist() for c in active},
        cooling_flags={c: case.cooling[c].astype(bool).tolist() for c in active},
        outdoor_temps={c: case.t_out[c].tolist() for c in active},
        sample_interval_s=case.panel.dt_seconds,
    )
    from_file = model.localise_leak(TEST_CASE)
    assert live["most_likely_faulty_car"] == from_file["most_likely_faulty_car"], (
        f"live {live['ranked_cars']} vs file {from_file['ranked_cars']}"
    )


# ---------------------------------------------------------------------------
# confidence: the bug this port exists to fix
# ---------------------------------------------------------------------------
def test_confidence_is_not_derived_from_the_margin():
    """
    ACV2 measured that healthy consists produce top-1 margins of the same size as
    faulty ones (median ratio 1.00x), so the margin cannot be the confidence
    signal. The old thresholds were margin >= 2.0 -> CLEAR, >= 0.5 -> MODERATE.
    """
    model = ACVSubsystemModel()
    res = model.localise_leak(TEST_CASE)

    assert res["margin_is_not_confidence"] is True
    assert res["confidence_level"] in ("strong", "moderate", "weak", "uncalibrated")
    assert res["confidence"] == CONFIDENCE_FROM_LEVEL[res["confidence_level"]]

    margin = res["confidence_margin"]
    # The test case has a large margin but weak calibrated separation: exactly
    # the disagreement that proves the two are not the same quantity.
    assert margin is not None and margin > 0.5
    assert res["confidence_level"] == "weak", (
        f"expected the test case to be weakly separated, got {res['confidence_level']}"
    )
    assert res["confidence"] == "AMBIGUOUS"


def test_confidence_is_referenced_to_a_fault_free_null():
    model = ACVSubsystemModel()
    res = model.localise_leak(TEST_CASE)
    assert res["null_calibrated"] is True
    assert res["n_null_consists"] >= 20
    assert 0.0 < res["null_p_value"] <= 1.0
    assert 0.0 <= res["separation_q"] <= 1.0


def test_detection_limit_is_reported_against_the_leading_evidence():
    model = ACVSubsystemModel()
    res = model.localise_leak(TEST_CASE)
    det = res["detection_limit"]
    assert det["evidence_k"] is not None
    assert det["reliable_from_k"] is not None
    # The test case's deficit is far below the magnitude at which recovery is
    # reliable; the response must say so rather than imply a firm diagnosis.
    assert det["below_detection_limit"] is True
    assert 0.0 <= det["expected_top1_rate"] <= 1.0


def test_weak_separation_becomes_watch_not_action_and_not_all_clear():
    """
    The old logic promoted anything with margin >= 0.5 to ACTION_NEEDED, and sent
    everything else to "all cars in thermal equilibrium". Both halves were wrong.
    A weak verdict must still name the suspect, but as a watch item.
    """
    model = ACVSubsystemModel()
    res = model.predict_from_csv(TEST_CASE, "acv_test_case.xlsx")

    assert res["confidence_level"] == "weak"
    assert res["status"] == "WATCH"
    assert res["verdict"] == "WATCH"
    assert res["most_likely_faulty_car"] == "01"
    assert res["recommended_action"] == "ACTION_INSPECT_ACV_PACK"
    # the suspect must be named, and the uncertainty stated
    assert "01" in res["conductor_summary"]
    assert "inconclusive" in res["conductor_summary"].lower()
    assert "all cars cooling in thermal equilibrium" not in res["conductor_summary"]


def test_competing_hypothesis_is_surfaced_to_the_operator():
    model = ACVSubsystemModel()
    res = model.localise_leak(TEST_CASE)
    contest = res["competing_hypothesis"]
    assert contest["alternative"] == res["ranked_cars_list"][1]
    assert contest["narrative"]
    against = {r["feature"] for r in contest["supports_alternative"]}
    assert "integrity_loss" in against
    for row in contest["supports_alternative"]:
        assert row["label"] and row["group"]


# ---------------------------------------------------------------------------
# live twin path
# ---------------------------------------------------------------------------
def test_live_twin_contract_unchanged():
    """The 10 Hz InferenceBroker path must keep its exact output contract."""
    model = ACVSubsystemModel()
    required = {
        "unit_id", "supply_temp_c", "return_temp_c", "delta_temp_c",
        "compressor_power_kw", "efficiency_rating", "anomaly_score", "fault_type",
    }

    healthy = model.evaluate_telemetry(19.2, 26.5, 4.82, degraded=False)
    assert required <= set(healthy)
    assert healthy["fault_type"] == "NONE"
    assert healthy["delta_temp_c"] == 7.3

    degraded = model.evaluate_telemetry(21.1, 26.5, 5.44, degraded=True)
    assert degraded["fault_type"] in ("REFRIGERANT_LEAKAGE", "FILTER_CLOGGING")
    # Degradation must actually degrade, or the What-If toggle is a dead control.
    assert degraded["efficiency_rating"] < healthy["efficiency_rating"]
    assert degraded["anomaly_score"] > healthy["anomaly_score"]


def test_no_model_artefact_required():
    """The model must construct and describe itself with no pickle present."""
    model = ACVSubsystemModel()
    meta = model.describe_method()
    assert meta["model_artefact"] is None
    assert meta["fitted_parameters"] == 0
    assert meta["requires_training_data"] is False
    assert meta["leak_localisation_available"] is True
    assert meta["implementation"] == "ACV2/acv2"
    # the demoted channel must be declared, not quietly carried
    assert "full_demand_frac" in meta["zero_weight_channels"]
    assert meta["confidence"]["statistic"] == "dixon_q_vs_fault_free_null"
    assert set(meta["evidence_groups"]) == set(acv2_cfg.FEATURE_GROUPS)


def test_ranking_works_without_the_confidence_artefacts():
    """
    The confidence artefacts are optional. With them absent the ranking must be
    unchanged and the level must degrade to 'uncalibrated' rather than a number
    being invented.
    """
    model = ACVSubsystemModel()
    baseline = model.localise_leak(TEST_CASE)

    bare = ACVSubsystemModel()
    bare._null = None
    bare._limit = None
    # re-route the module-level loaders the ranker consults
    original_null = acv2_confidence.load_null_calibration
    original_limit = acv2_confidence.load_detection_limit
    acv2_confidence.load_null_calibration = lambda path=None: None
    acv2_confidence.load_detection_limit = lambda path=None: None
    try:
        res = bare.localise_leak(TEST_CASE)
    finally:
        acv2_confidence.load_null_calibration = original_null
        acv2_confidence.load_detection_limit = original_limit

    assert res["ranked_cars"] == baseline["ranked_cars"], "ranking must not depend on calibration"
    assert res["confidence_level"] == "uncalibrated"
    assert res["confidence"] == "UNCALIBRATED"
    assert res["null_p_value"] is None
    assert res["null_calibrated"] is False


# ---------------------------------------------------------------------------
# upload contract
# ---------------------------------------------------------------------------
def _synthetic_consist_csv(leak_car="04", n=600, delta=1.6):
    """Build a consist CSV in the PS3 ACV header format, with a planted leak."""
    rng = np.random.default_rng(3)
    cols = {"Time": pd.date_range("2026-01-01", periods=n, freq="30s")}
    outdoor = 31.0 + rng.normal(0, 0.4, n)
    for i in range(1, 9):
        cid = f"{i:02d}"
        base = 24.0 + rng.normal(0, 0.05, n)
        if cid == leak_car:
            base = base + delta
        cols[f"Car {cid} - Indoor Average Temperature"] = base
        cols[f"Car {cid} - Outdoor Average Temperature"] = outdoor
        cols[f"Car {cid} - ACV Control Temperature (Cooling)"] = 24.0
        cols[f"Car {cid} - ACV Running Mode"] = "Automatic Cooling"
        cols[f"Car {cid} - ACV Information Valid"] = "Valid"
    return pd.DataFrame(cols).to_csv(index=False)


def test_upload_route_contract():
    """
    backend/api/routes.py calls predict_from_csv for user uploads. That path must
    work and must keep the shared cross-subsystem response contract.
    """
    model = ACVSubsystemModel()
    result = model.predict_from_csv(_synthetic_consist_csv(leak_car="04"),
                                    "acv_upload.csv")

    required = {
        "subsystem", "file_name", "status", "verdict", "anomaly_score",
        "most_likely_faulty_car", "confidence", "confidence_margin",
        "recommended_action", "conductor_summary", "ranked_cars", "car_diagnostics",
    }
    missing = required - set(result)
    assert not missing, f"upload response missing keys: {sorted(missing)}"
    assert result["subsystem"] == "acv"
    assert result["file_name"] == "acv_upload.csv"
    assert result["status"] in ("GOOD", "WATCH", "ACTION_NEEDED")
    assert result["most_likely_faulty_car"] == "04", (
        f"planted leak not found, got {result['most_likely_faulty_car']}"
    )
    assert isinstance(result["ranked_cars"], list) and len(result["ranked_cars"]) == 8


def test_a_clear_planted_leak_reaches_action_needed():
    """
    A large, unambiguous planted deficit must clear the calibrated bar, otherwise
    the new confidence gate would never fire and the feature would be dead.
    """
    model = ACVSubsystemModel()
    res = model.predict_from_csv(_synthetic_consist_csv(leak_car="06", delta=3.0),
                                 "acv_big_leak.csv")
    assert res["most_likely_faulty_car"] == "06"
    assert res["confidence_level"] in ACTIONABLE_LEVELS, (
        f"an obvious leak must be actionable, got {res['confidence_level']} "
        f"(Q={res['separation_q']}, p={res['null_p_value']})"
    )
    assert res["status"] == "ACTION_NEEDED"


def test_localise_leak_accepts_dataframe_and_path_alike():
    """A DataFrame and its CSV round-trip must score identically."""
    import io
    import tempfile
    model = ACVSubsystemModel()
    csv_text = _synthetic_consist_csv(leak_car="06")
    df = pd.read_csv(io.StringIO(csv_text))

    from_df = model.localise_leak(df, file_id="mem.csv")
    assert from_df["file_id"] == "mem.csv"
    assert from_df["most_likely_faulty_car"] == "06"

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "consist.csv")
        with open(p, "w", encoding="utf-8", newline="") as fh:
            fh.write(csv_text)
        from_path = model.localise_leak(p)
    assert from_path["ranked_cars"] == from_df["ranked_cars"]
    assert from_path["file_id"] == "consist.csv"


def test_batch_ranks_by_calibrated_evidence_not_margin():
    model = ACVSubsystemModel()
    files = [
        ("weak.csv", _synthetic_consist_csv(leak_car="02", delta=0.08).encode()),
        ("strong.csv", _synthetic_consist_csv(leak_car="06", delta=3.0).encode()),
    ]
    res = model.predict_batch(files, "acv_batch.zip")
    assert res["is_batch"] is True
    assert res["total_files"] == 2
    assert res["worst_file"] == "strong.csv", (
        f"batch picked {res['worst_file']}; it must rank by calibrated evidence"
    )
    assert res["most_likely_faulty_car"] == "06"
    for item in res["batch_items"]:
        assert "separation_q" in item and "confidence_level" in item


def test_rejects_unrecognised_schema():
    """A CSV with no ACV columns must fail loudly, not rank noise."""
    model = ACVSubsystemModel()
    junk = pd.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
    try:
        model.localise_leak(junk, file_id="junk.csv")
    except ValueError as exc:
        assert "junk.csv" in str(exc)
    else:
        raise AssertionError("expected ValueError for an unrecognised schema")


# ---------------------------------------------------------------------------
def _run_standalone() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  [FAIL] {fn.__name__}: {exc}")
        except Exception as exc:                                  # noqa: BLE001
            failed += 1
            print(f"  [ERROR] {fn.__name__}: {type(exc).__name__}: {exc}")
    print()
    print(f"  {len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    print("ACV subsystem model - parity and behaviour tests")
    print("=" * 70)
    raise SystemExit(_run_standalone())
