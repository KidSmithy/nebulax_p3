"""
acv2.config
===========================================================================
Static configuration for the ACV refrigerant-leak localisation system.

Everything that is a *convention* of the dataset (column spellings, mode
vocabularies) or a *physical constant / threshold* lives here, so the
modelling code never hardcodes a schema assumption.
===========================================================================
"""
from __future__ import annotations

import os

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
ACV2_DIR = os.path.dirname(PKG_DIR)
REPO_DIR = os.path.dirname(ACV2_DIR)

DATASET_DIR = os.path.join(REPO_DIR, "PS3", "02_Datasets", "ACV")
TRAIN_DIR = os.path.join(DATASET_DIR, "Train")
TEST_DIR = os.path.join(DATASET_DIR, "Test")
LABELS_CSV = os.path.join(DATASET_DIR, "Train_Labels.csv")

CACHE_DIR = os.path.join(ACV2_DIR, "cache")
ARTIFACT_DIR = os.path.join(ACV2_DIR, "artifacts")
REPORT_DIR = os.path.join(ACV2_DIR, "reports")

MODEL_PATH = os.path.join(ARTIFACT_DIR, "acv2_model.joblib")
# Calibration artefacts produced by scripts/validate_robustness.py. Both are
# optional: without them the ranker reports a raw separation statistic and
# simply declines to attach a calibrated probability to it.
NULL_CALIBRATION_PATH = os.path.join(ARTIFACT_DIR, "null_calibration.json")
DETECTION_LIMIT_PATH = os.path.join(ARTIFACT_DIR, "detection_limit.json")

# --------------------------------------------------------------------------
# Schema harmonisation
# --------------------------------------------------------------------------
# Per-car columns are named "Car <NN> - <parameter>". Different case files use
# different spellings for the same physical quantity; the loader maps the raw
# parameter name onto a canonical signal name using the table below.
#
# Matching is done on the *lower-cased* parameter text, longest pattern first,
# so that e.g. "Refrigeration System 1 Low Pressure Value" is matched before
# any shorter generic pattern.
CANONICAL_SIGNALS: dict[str, str] = {
    # ---- cabin / ambient temperatures -----------------------------------
    "indoor average temperature": "t_in",
    "passenger cabin temperature detected value": "t_in",
    "observation area temperature detected value": "t_in_obs",
    "outdoor average temperature": "t_out",
    "outside temperature sensor reading": "t_out",
    "fresh air temperature detected value": "t_out",
    # ---- control set points ---------------------------------------------
    "acv control temperature (cooling)": "t_set_cool",
    "acv control temperature (heating)": "t_set_heat",
    "target temperature value": "t_set_cool",
    # ---- discrete control / status --------------------------------------
    "acv running mode": "run_mode",
    "acv setting mode": "set_mode",
    "acv operating mode": "op_mode",
    "acv control mode": "ctrl_mode",
    "acv information valid": "info_valid",
    "acv self-check": "self_check",
    "load halved": "load_halved",
    "load shedding": "load_halved",
    # ---- refrigerant circuit (rich schema only) -------------------------
    "refrigeration system 1 low pressure value": "p_low_1",
    "refrigeration system 2 low pressure value": "p_low_2",
    "refrigeration system 1 high pressure value": "p_high_1",
    "refrigeration system 2 high pressure value": "p_high_2",
    "refrigeration system 1 energized solenoid valve open": "sv_1",
    "refrigeration system 2 energized solenoid valve open": "sv_2",
    "compressor 1 running": "comp_run_1",
    "compressor 2 running": "comp_run_2",
    "compressor 1 fault": "comp_fault_1",
    "compressor 2 fault": "comp_fault_2",
    "condenser fan 1 running": "cfan_run_1",
    "condenser fan 2 running": "cfan_run_2",
    "condenser fan 1 fault": "cfan_fault_1",
    "condenser fan 2 fault": "cfan_fault_2",
    "ventilation fan 1 running": "vfan_run_1",
    "ventilation fan 2 running": "vfan_run_2",
    "electric heater 1 running": "heater_run_1",
    "electric heater 2 running": "heater_run_2",
}

# Non-car identifying columns, matched case-insensitively on the full header.
TIME_COLUMN_HINTS = ("time", "timestamp", "date")
ID_COLUMNS = ("car model", "train number")

# --------------------------------------------------------------------------
# Mode vocabularies
# --------------------------------------------------------------------------
# Any running-mode string containing one of these means the refrigeration
# circuit is being asked to produce cooling.
COOLING_MODE_TOKENS = ("cooling", "refrigerat", "cool")
# Demand tier: how hard the controller is asking the pack to work. A pack that
# has lost charge cannot hold set point, so the controller escalates.
FULL_DEMAND_TOKENS = ("full cooling", "full refrigeration")
HALF_DEMAND_TOKENS = ("half cooling", "half refrigeration", "automatic cooling")
NON_OPERATING_TOKENS = ("stop", "stopped", "invalid", "self-check", "self check",
                        "ventilation", "off", "shutdown", "fault")
INVALID_TOKENS = ("invalid", "无效")

# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
CLEANING = {
    # Physically implausible cabin temperatures: sensor dropouts in this
    # dataset show up as exact 0.0 C or as values pinned at a rail.
    "t_in_min": 5.0,
    "t_in_max": 45.0,
    "t_out_min": -20.0,
    "t_out_max": 60.0,
    "t_set_min": 14.0,
    "t_set_max": 32.0,
    # Gauge pressures (kPa or 0.01 MPa depending on unit scaling) - only used
    # to reject obvious dropouts, never to threshold a diagnosis.
    "p_min": 1e-6,
    "p_max": 1e5,
    # A sampling gap larger than this many median intervals starts a new
    # continuous segment; rolling/derivative features never cross a segment.
    "gap_factor": 5.0,
    # Minimum rows a segment needs before it can contribute derivatives.
    "min_segment_rows": 6,
}

# --------------------------------------------------------------------------
# Physics / feature extraction
# --------------------------------------------------------------------------
PHYSICS = {
    # Rolling window used for the persistence and drift statistics, expressed
    # in *physical minutes* and converted to rows using each file's own dt.
    "persist_window_min": 15.0,
    # A peer-relative elevation above this is "warm" (degrees C). Chosen well
    # above the 0.1 C sensor quantisation and typical inter-car spread.
    "elevation_threshold_c": 0.20,
    # Load-sensitivity regression: the thermal load proxy is (t_out - t_set).
    # Bins below this population are dropped from the stratified estimate.
    "min_bin_rows": 20,
    "n_load_bins": 5,
    # Pull-down detection: a cooling ramp is a run of rows in cooling mode
    # where the smoothed cabin temperature is falling.
    "pulldown_smooth_min": 5.0,
    "pulldown_min_rows": 6,
    # CUSUM change detection on the peer-relative residual.
    "cusum_k_c": 0.15,      # slack (deg C) - drift smaller than this is noise
    "cusum_h_c": 6.0,       # decision threshold in accumulated deg C
    "cusum_clip_c": 1.0,    # per-step cap, so one sensor glitch cannot alarm
    # Robust z-score clipping, to stop one wild car dominating the fusion.
    "z_clip": 4.0,
}

# --------------------------------------------------------------------------
# Feature registry
# --------------------------------------------------------------------------
# Each entry: canonical feature name -> (group, orientation, default weight,
# human-readable description).
#
#   orientation = +1  -> larger value is more leak-like
#   orientation = -1  -> smaller value is more leak-like
#
# The default weights encode the physical prior (how diagnostic each signal is
# for undercharge). They are re-checked by leave-one-file-out cross validation
# in scripts/evaluate_loocv.py; calibration may rescale a group but the signs
# are fixed by physics and are never learned from the six labelled files.
FEATURE_SPEC: dict[str, tuple[str, int, float, str]] = {
    # ---- thermal branch: the capacity deficit itself ----------------------
    "elev_mean": ("thermal", +1, 1.00,
                  "Mean cabin temperature above the sibling-car median during cooling (K)"),
    "elev_load_stratified": ("thermal", +1, 1.00,
                             "Same elevation, averaged with equal weight over thermal-load "
                             "bins so an unequal duty history cannot fake it (K)"),
    "elev_p90": ("thermal", +1, 0.35,
                 "90th percentile elevation: worst-case capacity shortfall (K)"),
    "elev_persistence": ("thermal", +1, 0.50,
                         "Fraction of cooling time with a sustained 15-min elevation"),
    "setpoint_error": ("thermal", +1, 0.60,
                       "Mean (T_in - T_set) during cooling: control demand the pack "
                       "cannot meet (K)"),
    # ---- capacity branch: leak versus sensor bias -------------------------
    "load_sensitivity": ("capacity", +1, 0.70,
                         "d(elevation)/d(thermal load) (K/K): a capacity-limited car falls "
                         "further behind as the load rises, a biased sensor does not"),
    "capacity_shortfall": ("capacity", +1, 0.55,
                           "Median (T_in - T_set)/(T_out - T_set): dimensionless fraction of "
                           "the demanded cooling span that is not delivered"),
    "pulldown_rate": ("capacity", +1, 0.45,
                      "Peak sustained pull-down rate (K/h, negative): a starved evaporator "
                      "cannot pull the cabin down as fast"),
    "greybox_cool_rate": ("capacity", -1, 0.40,
                          "Identified cooling authority Q_max/C (K/h) from the grey-box "
                          "energy balance"),
    # ---- progression branch: a leak is monotonic -------------------------
    "elev_trend": ("progression", +1, 0.45,
                   "Trend of the peer-relative elevation over the record (K/day)"),
    "cusum_fraction": ("progression", +1, 0.35,
                       "Fraction of the record after a CUSUM change point on the residual"),
    # ---- controller response --------------------------------------------
    # DEMOTED TO ZERO WEIGHT. Physically this should be diagnostic - a pack that
    # cannot hold set point ought to be commanded to full cooling more often -
    # but measurement disagrees: on the six labelled files the channel scores
    # 0.531 used alone, *below* the 0.5625 random-permutation baseline, it is
    # computable in only 4 of 6 files, and its mean z at the true faulty car is
    # negative (-1.241), i.e. the faulty car is commanded to full cooling *less*
    # often than its siblings. The likely reason is that the demand tier is a
    # controller state that depends on the set point and the load schedule, not
    # on delivered capacity, so it tracks duty history rather than health.
    # It is kept computed and reported as a descriptive channel, but it is given
    # zero weight so it can never contribute noise to a ranking. Group ablation
    # confirms the training score is 1.0000 with the control group removed.
    "full_demand_frac": ("control", +1, 0.00,
                         "Fraction of cooling time the controller escalates to full cooling "
                         "(DESCRIPTIVE ONLY: scored 0.531 alone vs 0.5625 random baseline, "
                         "so it carries zero weight)"),
    # ---- refrigerant-circuit branch (rich schema only) -------------------
    "circuit_asym_lift": ("refrigerant", +1, 1.00,
                          "Relative mismatch in pressure lift (p_high - p_low) between the "
                          "pack's two circuits: the sibling circuit is a reference that "
                          "shares ambient, cabin and demand exactly"),
    "circuit_asym_high": ("refrigerant", +1, 0.70,
                          "Relative mismatch in condensing pressure between the two circuits"),
    "circuit_asym_ratio": ("refrigerant", +1, 0.50,
                           "Mismatch in compression ratio p_high/p_low between the two circuits"),
    "suction_excursion": ("refrigerant", +1, 0.80,
                          "Worst circuit's suction-pressure excursion (p50 - p05): "
                          "intermittent evaporator starvation"),
    "lift_deficit": ("refrigerant", +1, 0.60,
                     "Shortfall of the weakest circuit's demand-matched pressure lift "
                     "against the sibling cars"),
    "compressor_duty": ("refrigerant", +1, 0.30,
                        "Compressor run fraction during cooling: compensating for lost capacity"),
    "compressor_cycling": ("refrigerant", +1, 0.25,
                           "Compressor starts per hour: low-pressure short cycling"),
    # ---- data-integrity branch (secondary, non-thermodynamic evidence) ---
    "integrity_loss": ("integrity", +1, 0.45,
                       "Rate of invalid-status flags and implausible sensor readings this "
                       "pack reports: a unit in trouble drops out"),
}

FEATURE_GROUPS = ("thermal", "capacity", "progression", "control", "refrigerant", "integrity")

# Group-level multipliers applied on top of the per-feature weights. Set by
# calibration; the defaults are the physical prior.
GROUP_WEIGHTS = {
    "thermal": 1.00,
    "capacity": 1.00,
    "progression": 1.00,
    "control": 1.00,
    "refrigerant": 1.25,   # a direct pressure measurement outranks an inference
    "integrity": 1.00,
}

# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------
FUSION = {
    # Weight of the physics score vs the unsupervised outlier score.
    "w_physics": 0.85,
    "w_outlier": 0.15,
    # IsolationForest settings for the unsupervised member.
    "iso_n_estimators": 300,
    "iso_random_state": 20260918,
    # Cars with no usable telemetry cannot be diagnosed; they are ranked last
    # in a stable, documented order rather than dropped (the submission must
    # rank every car that appears in the file's headers).
    "inactive_penalty": -1e6,
}

# --------------------------------------------------------------------------
# Confidence
# --------------------------------------------------------------------------
# The raw top-1 margin is *not* interpretable on its own. Measurement:
# deleting the known faulty car from each labelled file and re-ranking the
# seven healthy siblings produces a top-1 margin of the same size (median
# margin ratio 0.94x, see reports/robustness.md test B). A large margin
# therefore does not mean a fault was found - it can simply mean one car is
# the warmest, which is true of every consist including a perfectly healthy
# one. Any confidence statement must be referenced to that null.
CONFIDENCE = {
    # Dixon-Q-style separation, scale free and standard for a single-outlier
    # test in a very small sample: (s1 - s2) / (s1 - sn).
    "statistic": "dixon_q",
    # Verdict bands on the null-referenced p-value: P(Q_null >= Q_observed)
    # estimated from the healthy-only consists of the labelled files.
    "p_strong": 0.05,
    "p_moderate": 0.20,
    # Below this many null samples the p-value is reported but flagged as
    # having too thin a calibration to lean on.
    "min_null_samples": 20,
    # A verdict whose leading evidence is smaller than the measured detection
    # limit is reported as provisional regardless of its separation, because
    # test C shows recovery is a coin flip in that regime.
    "reliable_top1_rate": 0.90,
}
