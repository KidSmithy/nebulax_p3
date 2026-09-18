"""
backend/models/test_acv_leak_model.py
===============================================================================
Parity and behaviour tests for the ACV subsystem model.

backend/models/acv_model.py carries an intentional standalone copy of the
scoring core in ACV/acv_physics_ranker.py. Duplicated code drifts silently, so
this test is the guard: it runs BOTH implementations over the real case files
and asserts they agree exactly, on every feature value and on the final
ranking.

It also checks the things that would quietly break the subsystem:
  - the live 10 Hz twin contract (evaluate_telemetry) still holds
  - the physics points the right way (a deliberately hot car must rank first)
  - the live-buffer path agrees with the file path on the same data

Run:  python -m pytest backend/models/test_acv_leak_model.py -q
      (or, with no pytest installed:  python backend/models/test_acv_leak_model.py)
===============================================================================
"""

from __future__ import annotations

import importlib.util
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

from acv_model import (  # noqa: E402
    ACVSubsystemModel,
    PHYSICS_WEIGHTS as BE_WEIGHTS,
    FEATURE_ORDER as BE_FEATURES,
    ROLE_ALIASES as BE_ALIASES,
    REL_ELEVATION_C as BE_REL,
    MIN_PEERS_COOLING as BE_PEERS,
    PERSISTENCE_MINUTES as BE_PERSIST,
    TEMP_MIN_C as BE_TMIN,
    TEMP_MAX_C as BE_TMAX,
)


def _load_reference():
    """Import ACV/acv_physics_ranker.py, the reference implementation."""
    path = os.path.join(ROOT, "ACV", "acv_physics_ranker.py")
    spec = importlib.util.spec_from_file_location("_acv_reference", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REF = _load_reference()


# ---------------------------------------------------------------------------
def test_constants_match_reference():
    """Every physics constant must be identical in both copies."""
    assert BE_WEIGHTS == REF.PHYSICS_WEIGHTS, "physics weights drifted"
    assert BE_FEATURES == REF.FEATURE_ORDER, "feature order drifted"
    assert BE_ALIASES == REF.ROLE_ALIASES, "schema aliases drifted"
    assert BE_REL == REF.REL_ELEVATION_C
    assert BE_PEERS == REF.MIN_PEERS_COOLING
    assert BE_PERSIST == REF.PERSISTENCE_MINUTES
    assert (BE_TMIN, BE_TMAX) == (REF.TEMP_MIN_C, REF.TEMP_MAX_C)


def test_ranking_matches_reference_on_real_cases():
    """
    Both implementations must produce byte-identical rankings and numerically
    identical features on the real telemetry.

    acv_case_04 is excluded only because it is 33 MB and slow to parse; the
    files used here cover both schema variants that matter at inference.
    """
    model = ACVSubsystemModel()
    cases = [TEST_CASE]
    for name in ("acv_case_01.xlsx", "acv_case_05.xlsx"):
        p = os.path.join(DATA, "Train", name)
        if os.path.exists(p):
            cases.append(p)

    assert cases, "no ACV case files found to test against"

    for path in cases:
        mine = model.localise_leak(path)
        theirs = REF.rank_consist(path)

        assert mine["ranked_cars"] == theirs["ranked_cars"], (
            f"{os.path.basename(path)}: ranking differs - "
            f"backend {mine['ranked_cars']} vs reference {theirs['ranked_cars']}"
        )
        assert mine["most_likely_faulty_car"] == theirs["most_likely_faulty_car"]
        assert mine["confidence_margin"] == theirs["confidence_margin"]
        assert mine["consist_size"] == theirs["consist_size"] == 8

        # Compare every feature value car by car.
        ref_feats = theirs["features"]
        for row in mine["car_diagnostics"]:
            if not row["assessable"]:
                continue
            c = row["car"]
            assert abs(row["mean_rel_c"] - round(float(ref_feats.loc[c, "mean_rel_cooling"]), 3)) < 1e-9
            assert abs(row["persistence_pct"]
                       - round(float(ref_feats.loc[c, "persistence_15m"]) * 100, 1)) < 1e-9
            assert row["n_cooling_samples"] == int(ref_feats.loc[c, "n_cooling_samples"])


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
    assert res["confidence"] in ("CLEAR", "MODERATE")


def test_live_window_agrees_with_file_path():
    """
    The live-buffer path and the file path share one scoring core, so feeding
    the file's own cleaned frames through rank_consist_window must reproduce
    the file-based ranking.
    """
    model = ACVSubsystemModel()
    from acv_model import (
        load_case, clean_indoor_temperatures, cooling_mask, setpoint_frame,
    )

    df, colmap, all_ids, tcol = load_case(TEST_CASE)
    indoor = clean_indoor_temperatures(df, colmap, all_ids)
    idx = indoor.index
    cool = cooling_mask(df, colmap, all_ids, idx)
    sp = setpoint_frame(df, colmap, all_ids, idx)
    active = [c for c in indoor.columns if indoor[c].notna().any()]

    live = model.rank_consist_window(
        cabin_temps={c: indoor[c].tolist() for c in active},
        setpoints={c: sp[c].tolist() for c in active},
        cooling_flags={c: cool[c].tolist() for c in active},
        sample_interval_s=30.0,
    )
    from_file = model.localise_leak(TEST_CASE)
    assert live["ranked_cars"] == from_file["ranked_cars"], (
        f"live {live['ranked_cars']} vs file {from_file['ranked_cars']}"
    )


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
    assert meta["physics_weights"] == dict(BE_WEIGHTS)


def _synthetic_consist_csv(leak_car="04", n=300):
    """Build a consist CSV in the PS3 ACV header format, with a planted leak."""
    rng = np.random.default_rng(3)
    cols = {"Time": pd.date_range("2026-01-01", periods=n, freq="30s")}
    for i in range(1, 9):
        cid = f"{i:02d}"
        base = 24.0 + rng.normal(0, 0.05, n)
        if cid == leak_car:
            base = base + 1.1
        cols[f"Car {cid} - Indoor Average Temperature"] = base
        cols[f"Car {cid} - ACV Control Temperature (Cooling)"] = 24.0
        cols[f"Car {cid} - ACV Running Mode"] = "Automatic Cooling"
        cols[f"Car {cid} - ACV Information Valid"] = "Valid"
    return pd.DataFrame(cols).to_csv(index=False)


def test_upload_route_contract():
    """
    backend/api/routes.py calls predict_from_csv for user uploads, which calls
    localise_leak with a DataFrame rather than a path. That path must work and
    must keep the shared cross-subsystem response contract.
    """
    model = ACVSubsystemModel()
    csv_text = _synthetic_consist_csv(leak_car="04")

    result = model.predict_from_csv(csv_text, "acv_upload.csv")

    required = {
        "subsystem", "file_name", "status", "verdict", "anomaly_score",
        "most_likely_faulty_car", "confidence", "confidence_margin",
        "recommended_action", "conductor_summary", "ranked_cars", "car_diagnostics",
    }
    missing = required - set(result)
    assert not missing, f"upload response missing keys: {sorted(missing)}"
    assert result["subsystem"] == "acv"
    assert result["file_name"] == "acv_upload.csv"
    assert result["most_likely_faulty_car"] == "04", (
        f"planted leak not found, got {result['most_likely_faulty_car']}"
    )
    assert isinstance(result["ranked_cars"], list) and len(result["ranked_cars"]) == 8


def test_localise_leak_accepts_dataframe_and_path_alike():
    """A DataFrame and its CSV round-trip must score identically."""
    import io
    model = ACVSubsystemModel()
    csv_text = _synthetic_consist_csv(leak_car="06")
    df = pd.read_csv(io.StringIO(csv_text))

    from_df = model.localise_leak(df, file_id="mem.csv")
    assert from_df["file_id"] == "mem.csv"
    assert from_df["most_likely_faulty_car"] == "06"

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "consist.csv")
        with open(p, "w", encoding="utf-8", newline="") as fh:
            fh.write(csv_text)
        from_path = model.localise_leak(p)
    assert from_path["ranked_cars"] == from_df["ranked_cars"]
    assert from_path["file_id"] == "consist.csv"


def test_rejects_unrecognised_schema():
    """A CSV with no ACV columns must fail loudly, not rank noise."""
    model = ACVSubsystemModel()
    junk = pd.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
    try:
        model.localise_leak(junk, file_id="junk.csv")
    except ValueError as exc:
        assert "no recognised ACV columns" in str(exc)
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
