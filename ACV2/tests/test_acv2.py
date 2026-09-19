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
# zero-weight demotion (remark 3)
# --------------------------------------------------------------------------
def test_below_baseline_channel_carries_zero_weight():
    """
    full_demand_frac scored 0.531 used alone against a 0.5625 random baseline,
    so it must not influence a ranking. It is still computed, because it is
    descriptive, but its weight is zero.
    """
    assert cfg.FEATURE_SPEC["full_demand_frac"][2] == 0.0
    features, _ = extract(clean(load_case(TEST_FILE)))
    assert features["full_demand_frac"].notna().any()      # still computed

    result = fuse(features, group_weights=DEFAULT_MODEL["group_weights"],
                  feature_weights=DEFAULT_MODEL["feature_weights"])
    assert "full_demand_frac" not in result["weights"]
    assert float(result["contributions"]["full_demand_frac"].abs().max()) == 0.0


def test_zero_weight_channel_cannot_reach_the_score_via_the_outlier_member():
    """
    The unsupervised member must see exactly the channels the physics member is
    allowed to use, otherwise "zero weight" would be a lie: IsolationForest
    would still consume the demoted channel and move the ranking.
    """
    features, _ = extract(clean(load_case(TEST_FILE)))
    baseline = fuse(features, group_weights=DEFAULT_MODEL["group_weights"],
                    feature_weights=DEFAULT_MODEL["feature_weights"])["score"]

    perturbed = features.copy()
    # Drive the demoted channel to an extreme on a car that is not the leader.
    perturbed.loc[perturbed.index[-1], "full_demand_frac"] = 1e3
    after = fuse(perturbed, group_weights=DEFAULT_MODEL["group_weights"],
                 feature_weights=DEFAULT_MODEL["feature_weights"])["score"]
    pd.testing.assert_series_equal(baseline, after)


def test_zero_weight_group_is_not_advertised_as_evidence():
    _features, ctx = extract(clean(load_case(TEST_FILE)))
    assert "control" not in ctx["active_groups"]
    assert "control" in ctx["descriptive_only_groups"]


def test_stale_artefact_cannot_override_a_deliberate_demotion():
    from acv2.ranker import _stale_weights
    stale = {"feature_weights": dict(DEFAULT_MODEL["feature_weights"],
                                     full_demand_frac=0.30)}
    assert _stale_weights(stale) == ["full_demand_frac"]
    assert _stale_weights({"feature_weights": DEFAULT_MODEL["feature_weights"]}) == []


# --------------------------------------------------------------------------
# confidence: separation must be referenced to a fault-free null
# --------------------------------------------------------------------------
def test_dixon_q_is_scale_free():
    """
    The raw margin depends on the arbitrary units of a weighted z-score pool,
    which is why it could not be compared between files. Q does not.
    """
    from acv2.confidence import dixon_q
    scores = pd.Series({"a": 3.0, "b": 1.0, "c": 0.5, "d": -1.0})
    assert abs(dixon_q(scores) - dixon_q(scores * 17.0)) < 1e-12
    assert abs(dixon_q(scores) - dixon_q(scores + 100.0)) < 1e-12
    assert 0.0 <= dixon_q(scores) <= 1.0
    assert np.isnan(dixon_q([1.0, 2.0]))                    # undefined below n=3
    assert np.isnan(dixon_q([2.0, 2.0, 2.0]))               # no spread


def test_separation_identifies_the_leader_and_the_alternative():
    from acv2.confidence import separation
    sep = separation(pd.Series({"01": 2.2, "04": 0.6, "03": 0.5, "05": -1.0}))
    assert sep["top_car"] == "01" and sep["runner_up"] == "04"
    assert sep["n_cars"] == 4
    assert abs(sep["margin"] - 1.6) < 1e-9


def test_null_p_value_is_bounded_and_never_exactly_zero():
    from acv2.confidence import null_p_value
    null = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert null_p_value(1.0, null) == 1 / 6                 # +1 correction
    assert null_p_value(0.0, null) == 1.0
    assert 0.0 < null_p_value(0.35, null) <= 1.0
    assert np.isnan(null_p_value(0.5, []))


def test_confidence_degrades_gracefully_without_a_calibration_artefact():
    from acv2.confidence import assess
    out = assess(pd.Series({"a": 3.0, "b": 1.0, "c": 0.0}), null={}, limit=None)
    assert out["level"] == "uncalibrated"
    assert np.isnan(out["null_p_value"])


def test_combined_p_value_pays_for_testing_two_statistics():
    """
    Q and top_z are complementary - measured on the six labelled files, Q catches
    cases 03 and 06 while top_z catches 01 and 02 - so both are tested and the
    smaller p-value is taken with a Bonferroni factor of two. The correction must
    be applied, and it must never manufacture significance.
    """
    from acv2.confidence import N_SEPARATION_TESTS, combined_p_value, p_value_for
    assert N_SEPARATION_TESTS == 2
    assert combined_p_value(0.02, 0.40) == 0.04        # 2 x the smaller
    assert combined_p_value(0.40, 0.02) == 0.04        # order must not matter
    assert combined_p_value(0.80, 0.90) == 1.0         # clamped, never > 1
    # A missing second statistic must not be treated as a significant one.
    assert combined_p_value(0.10, float("nan")) == 0.20
    assert np.isnan(combined_p_value(float("nan"), float("nan")))

    null = {"dixon_q_null": [0.1] * 10, "top_z_null": [1.0] * 10}
    both = p_value_for({"dixon_q": 0.9, "top_z": 9.0}, null)
    assert both == 2 * (1 / 11)
    assert np.isnan(p_value_for({"dixon_q": 0.9, "top_z": 9.0}, None))


def test_confidence_is_downgraded_below_the_measured_detection_limit():
    """
    Separation and detectability are different questions. A verdict resting on a
    deficit smaller than the magnitude at which recovery becomes reliable must
    not be reported as strong however cleanly it separates.
    """
    from acv2.confidence import assess
    # Both nulls must be supplied, or the Bonferroni factor doubles a p-value
    # without a second test having been paid for.
    null = {"dixon_q_null": [0.02 * i for i in range(40)],      # max 0.78
            "top_z_null": [0.2 * i for i in range(40)]}          # max 7.8
    limit = {"curve": [{"delta_k": 0.05, "top1_rate": 0.22, "trials": 18},
                       {"delta_k": 1.00, "top1_rate": 1.00, "trials": 18}]}
    scores = pd.Series({"a": 10.0, "b": 0.1, "c": 0.0})

    clean_big = assess(scores, evidence_k=2.0, null=null, limit=limit)
    assert clean_big["level"] == "strong", (
        f"expected strong, got {clean_big['level']} "
        f"(p={clean_big['null_p_value']}, p_q={clean_big['null_p_dixon_q']}, "
        f"p_z={clean_big['null_p_top_z']})")
    assert clean_big["downgraded_for"] is None

    clean_small = assess(scores, evidence_k=0.11, null=null, limit=limit)
    assert clean_small["level"] == "moderate"
    assert clean_small["downgraded_for"] == "below_detection_limit"
    assert clean_small["detection"]["below_detection_limit"] is True
    assert clean_small["detection"]["expected_top1_rate"] == 0.22


def test_ranker_reports_calibrated_confidence_not_a_raw_margin():
    result = rank_case(load_case(TEST_FILE))
    conf = result["confidence"]
    assert conf["level"] in {"strong", "moderate", "weak", "uncalibrated"}
    assert conf["top_car"] == result["most_likely_faulty_car"]
    # the verdict text must not present the bare margin as confidence
    assert "margin" not in result["verdict"].lower()
    if np.isfinite(conf["null_p_value"]):
        assert "Confidence:" in result["verdict"]


# --------------------------------------------------------------------------
# competing hypothesis (remark 4)
# --------------------------------------------------------------------------
def test_competing_hypothesis_names_the_runner_up_and_its_evidence():
    result = rank_case(load_case(TEST_FILE))
    contest = result["competing_hypothesis"]
    ranked = result["ranked_cars_list"]
    assert contest["alternative"] == ranked[1]
    assert contest["supports_leader"], "the leader must have named supporting evidence"
    # every listed channel must point the way the sign of its delta claims
    for row in contest["supports_leader"]:
        assert row["delta"] > 0
    for row in contest["supports_alternative"]:
        assert row["delta"] < 0
    assert 0.0 <= contest["evidence_share_against"] <= 1.0


def test_split_evidence_is_reported_rather_than_hidden():
    """
    On the test case the integrity channel favours the runner-up while the
    thermal and capacity channels favour the leader. The report must say so.
    """
    result = rank_case(load_case(TEST_FILE))
    contest = result["competing_hypothesis"]
    against = {row["feature"] for row in contest["supports_alternative"]}
    assert "integrity_loss" in against
    assert contest["narrative"]
    assert contest["alternative"] in contest["narrative"]


# --------------------------------------------------------------------------
# car-level bootstrap (remark 5)
# --------------------------------------------------------------------------
def test_without_cars_shrinks_the_consist_and_recomputes_the_reference():
    case = clean(load_case(THIN_FILE))
    smaller = case.without_cars([case.cars[0], case.cars[1]])
    assert len(smaller.cars) == len(case.cars) - 2
    assert case.cars[0] not in smaller.header_cars
    assert case.cars[0] not in smaller.t_in.columns
    # the peer reference genuinely changes, which is the point of the test
    features, _ = extract(smaller)
    assert len(features) == len(smaller.cars)


def test_peer_dropout_bootstrap_tests_a_different_axis_from_time_blocks():
    from acv2.evaluate import peer_dropout_stability
    case = clean(load_case(THIN_FILE))
    stab = peer_dropout_stability(case, DEFAULT_MODEL, n_draws=4, drop_k=2)
    assert stab["n_draws"] > 0
    assert stab["baseline_top"] in case.cars
    assert 0.0 <= stab["leader_retained"] <= 1.0
    # each draw ranked a smaller consist than the original
    assert all(0.0 <= f <= 1.0 for f in stab["top1_frequency"].values())


# --------------------------------------------------------------------------
# refrigerant branch audit (remark 2)
# --------------------------------------------------------------------------
def test_refrigerant_branch_is_auditable_against_the_inferred_branch():
    """
    The branch is computable in one file only. Rather than downweight the most
    physically direct evidence for an accident of instrumentation, the two
    branches are scored separately so the disagreement is on the record.
    """
    from acv2.evaluate import INFERRED_GROUPS, _branch_model
    features, ctx = extract(clean(load_case(RICH_FILE)))
    assert ctx["has_refrigerant"] is True

    inferred = fuse(features, **{k: v for k, v in _branch_model(INFERRED_GROUPS).items()
                                 if k in ("group_weights", "feature_weights",
                                          "w_physics", "w_outlier")})
    circuit = fuse(features, **{k: v for k, v in _branch_model(("refrigerant",)).items()
                                if k in ("group_weights", "feature_weights",
                                         "w_physics", "w_outlier")})
    inferred_top = inferred["score"].idxmax()
    circuit_top = circuit["score"].idxmax()
    true_car = load_labels()["acv_case_04.xlsx"]
    # the documented finding: the circuit branch is what gets this file right
    assert circuit_top == true_car
    assert inferred_top != true_car


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
