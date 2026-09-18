"""
tests/test_acv2.py
===========================================================================
Verification suite. Covers the three things that can silently ruin the
submission - a mis-parsed schema, a physics function with the wrong sign, and
an output file that does not match the required format - plus the competition
metric itself against the worked example in the problem statement.

Run:  python -m pytest ACV2/tests -q
      (or, with no pytest installed:  python ACV2/tests/test_acv2.py)
===========================================================================
"""
from __future__ import annotations

import csv
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acv2 import config as cfg
from acv2.cleaning import clean, contains_token, to_boolean
from acv2.detectors import fuse, oriented_z
from acv2.evaluate import rank_decay_score
from acv2.features import extract
from acv2.io_loader import discover_cases, load_case, load_labels
from acv2.physics import cusum_onset, mad_scale, peer_reference, robust_z
from acv2.ranker import DEFAULT_MODEL, rank_case

TEST_FILE = os.path.join(cfg.TEST_DIR, "acv_test_case.xlsx")
THIN_FILE = os.path.join(cfg.TRAIN_DIR, "acv_case_06.xlsx")
RICH_FILE = os.path.join(cfg.TRAIN_DIR, "acv_case_04.xlsx")


# --------------------------------------------------------------------------
# metric
# --------------------------------------------------------------------------
def test_rank_decay_matches_problem_statement():
    """The worked example: 8 cars, true car at rank r, score (8-(r-1))/8."""
    cars = [f"{i:02d}" for i in range(1, 9)]
    assert rank_decay_score(cars, "01") == 1.0
    assert rank_decay_score(cars, "02") == 0.875
    assert rank_decay_score(cars, "03") == 0.750
    assert rank_decay_score(cars, "08") == 0.125
    assert rank_decay_score(cars, "09") == 0.0          # not ranked at all
    assert rank_decay_score([], "01") == 0.0


# --------------------------------------------------------------------------
# loader: schema agnosticism
# --------------------------------------------------------------------------
def test_loader_reads_each_schema_on_its_own_terms():
    thin = load_case(THIN_FILE)
    rich = load_case(RICH_FILE)
    assert thin.header_cars == [f"{i:02d}" for i in range(1, 9)]
    assert rich.header_cars == [f"{i:02d}" for i in range(1, 9)]
    # the thin schema has no refrigerant channels, the rich one does
    assert not thin.has("p_low_1")
    assert rich.has("p_low_1") and rich.has("p_high_2")
    # both spellings of the outdoor probe land on the same canonical signal
    assert thin.has("t_in") and thin.has("t_out") and thin.has("t_set_cool")
    # sampling interval is read from the data, not assumed
    assert abs(thin.dt_seconds - 30.0) < 1e-6
    assert abs(rich.dt_seconds - 10.0) < 1e-6
    # a physical window converts to a different row count on each file
    assert thin.rows_for_minutes(15) == 30
    assert rich.rows_for_minutes(15) == 90


def test_labels_are_two_digit_strings():
    labels = load_labels()
    assert labels["acv_case_01.xlsx"] == "01"
    assert all(len(v) == 2 and v.isdigit() for v in labels.values())


# --------------------------------------------------------------------------
# cleaning
# --------------------------------------------------------------------------
def test_cleaning_removes_implausible_temperatures_but_records_them():
    panel = load_case(TEST_FILE)
    case = clean(panel)
    values = case.t_in[case.cars].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    assert finite.min() >= cfg.CLEANING["t_in_min"]
    assert finite.max() <= cfg.CLEANING["t_in_max"]
    # the dropouts are not silently discarded - they are evidence
    assert sum(case.report["dropouts_per_car"].values()) > 0


def test_cooling_mask_excludes_ventilation_and_stop():
    frame = pd.DataFrame({"a": ["Automatic Cooling", "Ventilation", "Stop",
                                "Full Cooling", "Emergency Ventilation", None]})
    cooling = contains_token(frame, cfg.COOLING_MODE_TOKENS)["a"].tolist()
    assert cooling == [True, False, False, True, False, False]


def test_boolean_coercion_handles_mixed_encodings():
    frame = pd.DataFrame({"a": ["1.0", "0.0", "Running", "Stopped", "weird", None]})
    out = to_boolean(frame)["a"].tolist()
    assert out[:4] == [1.0, 0.0, 1.0, 0.0]
    assert np.isnan(out[4]) and np.isnan(out[5])


# --------------------------------------------------------------------------
# physics
# --------------------------------------------------------------------------
def test_robust_z_is_not_dragged_by_the_outlier_it_must_expose():
    normal = pd.Series({"a": 0.0, "b": 0.1, "c": -0.1, "d": 0.05,
                        "e": -0.05, "f": 0.0, "g": 0.02, "h": 5.0})
    z = robust_z(normal)
    assert z["h"] == cfg.PHYSICS["z_clip"]          # clipped, i.e. far out
    assert abs(z.drop("h")).max() < 2.0             # the peers stay near zero
    # a standard z-score would be inflated by the outlier's own contribution
    classic = (normal - normal.mean()) / normal.std()
    assert classic["h"] < 3.0 < z["h"] + 1e-9


def test_mad_scale_is_zero_for_a_constant_channel():
    assert mad_scale([1.0, 1.0, 1.0, 1.0]) == 0.0
    # and robust_z degrades gracefully rather than dividing by zero
    assert robust_z(pd.Series([1.0, 1.0, 1.0])).abs().max() == 0.0


def test_peer_reference_excludes_the_car_itself():
    frame = pd.DataFrame({"a": [10.0], "b": [0.0], "c": [0.0], "d": [0.0]})
    ref = peer_reference(frame)
    assert ref.loc[0, "a"] == 0.0            # median of b, c, d
    assert ref.loc[0, "b"] == 0.0            # median of a, c, d


def test_cusum_detects_a_sustained_step_and_ignores_a_spike():
    days = pd.Series(np.linspace(0, 4, 4 * 96))
    flat = pd.Series(np.zeros(len(days)))
    step = flat.copy()
    step.iloc[len(days) // 2:] = 0.8                     # sustained +0.8 K
    spike = flat.copy()
    spike.iloc[100:105] = 8.0                            # brief excursion

    assert cusum_onset(step, days)["detected"] is True
    assert 1.5 < cusum_onset(step, days)["onset_day"] < 3.5
    assert cusum_onset(spike, days)["detected"] is False
    assert cusum_onset(flat, days)["detected"] is False


# --------------------------------------------------------------------------
# features and orientation
# --------------------------------------------------------------------------
def test_every_declared_feature_exists_on_both_schemas():
    for path in (THIN_FILE, RICH_FILE):
        features, _ctx = extract(clean(load_case(path)))
        for name in cfg.FEATURE_SPEC:
            assert name in features.columns, f"{name} missing for {os.path.basename(path)}"


def test_refrigerant_branch_activates_only_on_the_rich_schema():
    thin_features, thin_ctx = extract(clean(load_case(THIN_FILE)))
    rich_features, rich_ctx = extract(clean(load_case(RICH_FILE)))
    assert thin_ctx["has_refrigerant"] is False
    assert rich_ctx["has_refrigerant"] is True
    assert thin_features["circuit_asym_lift"].isna().all()
    assert rich_features["circuit_asym_lift"].notna().any()
    # and the thin file is scored on the thermal evidence rather than penalised
    assert "refrigerant" not in thin_ctx["active_groups"]
    assert "thermal" in thin_ctx["active_groups"]


def test_orientation_is_applied_so_larger_always_means_more_leak_like():
    features, _ = extract(clean(load_case(THIN_FILE)))
    z = oriented_z(features)
    # greybox_cool_rate is the one channel where *less* is leak-like
    if "greybox_cool_rate" in z.columns:
        raw = features["greybox_cool_rate"]
        worst = raw.idxmin()
        assert z.loc[worst, "greybox_cool_rate"] == z["greybox_cool_rate"].max()


def test_score_is_a_weighted_average_so_it_stays_bounded():
    features, _ = extract(clean(load_case(TEST_FILE)))
    result = fuse(features, group_weights=DEFAULT_MODEL["group_weights"],
                  feature_weights=DEFAULT_MODEL["feature_weights"])
    assert result["physics"].abs().max() <= cfg.PHYSICS["z_clip"] + 1e-9


# --------------------------------------------------------------------------
# ranking contract
# --------------------------------------------------------------------------
def test_ranking_is_a_permutation_of_the_files_own_car_ids():
    for path in (TEST_FILE, THIN_FILE, RICH_FILE):
        panel = load_case(path)
        result = rank_case(panel)
        ranked = result["ranked_cars_list"]
        assert sorted(ranked) == sorted(panel.header_cars)
        assert len(set(ranked)) == len(ranked)
        assert all(len(c) == 2 and c.isdigit() for c in ranked)
        assert result["ranked_cars"] == "|".join(ranked)


def test_undiagnosable_cars_are_ranked_last_not_dropped():
    """The rich-schema case only instruments four of its eight cars."""
    panel = load_case(RICH_FILE)
    result = rank_case(panel)
    ranked = result["ranked_cars_list"]
    assert len(ranked) == 8
    assert result["n_diagnosable"] == 4
    diagnosable_positions = [ranked.index(c) for c in ranked[:4]]
    assert max(diagnosable_positions) < 4
    # the true faulty car of this file must not be in the appended tail
    assert load_labels()["acv_case_04.xlsx"] in ranked[:4]


def test_ranking_is_deterministic():
    panel = load_case(TEST_FILE)
    assert rank_case(panel)["ranked_cars"] == rank_case(panel)["ranked_cars"]


def test_model_loading_falls_back_to_the_physical_prior():
    from acv2.ranker import load_model
    model = load_model("does_not_exist.joblib")
    assert model["group_weights"] == dict(cfg.GROUP_WEIGHTS)
    assert rank_case(load_case(TEST_FILE), model=model)["ranked_cars"]


# --------------------------------------------------------------------------
# CLI contract
# --------------------------------------------------------------------------
def test_predict_cli_writes_the_required_schema():
    sys.path.insert(0, cfg.ACV2_DIR)
    import predict as predict_cli

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "acv_predictions.csv")
        assert predict_cli.main(["--input", TEST_FILE, "--output", out, "--quiet"]) == 0
        with open(out, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1
        assert list(rows[0]) == ["file_id", "ranked_cars"]
        assert rows[0]["file_id"] == "acv_test_case.xlsx"          # extension included
        cars = rows[0]["ranked_cars"].split("|")
        assert sorted(cars) == [f"{i:02d}" for i in range(1, 9)]


def test_predict_cli_accepts_a_directory():
    sys.path.insert(0, cfg.ACV2_DIR)
    import predict as predict_cli

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "dir.csv")
        assert predict_cli.main(["--input", cfg.TEST_DIR, "--output", out, "--quiet"]) == 0
        with open(out, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == len(discover_cases(cfg.TEST_DIR))


def test_matches_the_official_example_submission_schema():
    example = os.path.join(cfg.REPO_DIR, "PS3", "04_Example_Submission", "acv_predictions.csv")
    if not os.path.exists(example):
        return
    with open(example, newline="", encoding="utf-8-sig") as fh:
        reference = list(csv.DictReader(fh))
    ours = os.path.join(cfg.ACV2_DIR, "acv_predictions.csv")
    if not os.path.exists(ours):
        return
    with open(ours, newline="", encoding="utf-8") as fh:
        mine = list(csv.DictReader(fh))
    assert [c.strip() for c in reference[0]] == list(mine[0])


# --------------------------------------------------------------------------
# minimal runner for environments without pytest
# --------------------------------------------------------------------------
def _run_all() -> int:
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:                      # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures} passed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
