"""
train.py
===========================================================================
Fit and export the ACV2 model artefact.

"Training" here means calibrating six group-level weight multipliers by
leave-one-file-out-validated coordinate ascent, and recording the evidence that
justifies the shipped weights. There is deliberately no fitted classifier: with
six labelled files and one faulty car each, a learned decision boundary would
memorise the training set, while the peer-consensus physics transfers to any
consist of any size on either schema.

The artefact is therefore small and auditable - weights, provenance and the
measured scores - and `predict.py` runs correctly even if it is absent.

Run:  python ACV2/scripts/train.py
Writes: ACV2/artifacts/acv2_model.joblib, ACV2/artifacts/train_features.csv
===========================================================================
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import datetime, timezone

import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acv2 import __version__                                     # noqa: E402
from acv2 import config as cfg                                   # noqa: E402
from acv2 import confidence                                      # noqa: E402
from acv2.evaluate import (RANDOM_BASELINE_8, build_case_set,     # noqa: E402
                           calibrate_groups, loocv, observed_separation)
from acv2.ranker import DEFAULT_MODEL                             # noqa: E402

warnings.filterwarnings("ignore")


def main() -> None:
    os.makedirs(cfg.ARTIFACT_DIR, exist_ok=True)
    case_set = build_case_set(verbose=True)

    print("\nScoring the uncalibrated physical prior ...")
    prior = case_set.score(DEFAULT_MODEL)
    prior_mean = float(prior["score"].mean())
    print(prior.to_string(index=False))
    print(f"  prior mean rank-decay score = {prior_mean:.4f}")

    print("\nCalibrating group weights (coordinate ascent on all labelled files) ...")
    groups, fitted = calibrate_groups(case_set, case_set.labelled(), verbose=True)
    print(f"  calibrated group weights = {groups}  (train score {fitted:.4f})")

    print("\nLeave-one-file-out cross validation of the whole procedure ...")
    cv = loocv(case_set)
    cv_mean = float(cv["score"].mean())
    print(f"  LOOCV mean rank-decay score = {cv_mean:.4f}")

    model = {
        "version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "group_weights": groups,
        "feature_weights": {name: spec[2] for name, spec in cfg.FEATURE_SPEC.items()},
        "feature_orientation": {name: spec[1] for name, spec in cfg.FEATURE_SPEC.items()},
        "w_physics": cfg.FUSION["w_physics"],
        "w_outlier": cfg.FUSION["w_outlier"],
        "calibrated": groups != dict(cfg.GROUP_WEIGHTS),
        "training_files": case_set.labelled(),
        "training_labels": {f: case_set.labels[f] for f in case_set.labelled()},
        "scores": {
            "prior_mean": prior_mean,
            "calibrated_train_mean": fitted,
            "loocv_mean": cv_mean,
            "random_baseline_8car": RANDOM_BASELINE_8,
            "per_file": prior.to_dict(orient="records"),
            "loocv_per_file": cv.drop(columns=["group_weights"]).to_dict(orient="records"),
        },
        "notes": ("Peer-consensus physics ranker. No fitted estimator: feature signs come "
                  "from the vapour-compression energy balance, weights from the physical "
                  "prior, and calibration only rescales six group multipliers."),
        "zero_weight_channels": {
            name: spec[3] for name, spec in cfg.FEATURE_SPEC.items() if spec[2] <= 0
        },
        "confidence_calibration": {
            "null_calibration_present": os.path.exists(cfg.NULL_CALIBRATION_PATH),
            "detection_limit_present": os.path.exists(cfg.DETECTION_LIMIT_PATH),
            "n_null_consists": len((confidence.load_null_calibration() or {})
                                   .get("dixon_q_null", [])),
            "built_by": "scripts/validate_robustness.py",
            "note": ("Confidence is referenced to fault-free consists, not to the raw top-1 "
                     "margin: measurement showed healthy consists separate as strongly as "
                     "faulty ones, so the margin is not evidence of fault presence."),
        },
    }
    joblib.dump(model, cfg.MODEL_PATH)
    print(f"\nWrote {cfg.MODEL_PATH}")

    print("\nSeparation of each labelled file against the fault-free null ...")
    separation = observed_separation(case_set, model)
    calibration = confidence.load_null_calibration()
    if calibration:
        separation["null_p_value"] = [
            confidence.null_p_value(q, calibration["dixon_q_null"])
            for q in separation["dixon_q"]]
    else:
        print("  (no null calibration yet - run scripts/validate_robustness.py)")
    print(separation.to_string(index=False))
    model["scores"]["separation"] = separation.to_dict(orient="records")

    # a flat feature dump, so the numbers behind every verdict are inspectable
    frames = []
    for file_id, features in case_set.features.items():
        block = features.copy()
        block.insert(0, "file_id", file_id)
        block.insert(1, "car", block.index)
        block.insert(2, "is_faulty", [int(c == case_set.labels.get(file_id)) for c in block.index])
        frames.append(block.reset_index(drop=True))
    dump_path = os.path.join(cfg.ARTIFACT_DIR, "train_features.csv")
    pd.concat(frames, ignore_index=True).to_csv(dump_path, index=False)
    print(f"Wrote {dump_path}")

    summary_path = os.path.join(cfg.ARTIFACT_DIR, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(model["scores"], fh, indent=2)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
