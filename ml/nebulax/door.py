from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .common import SEED, finite_frame, validate_submission, write_json
from .metrics import door_exact_boundary_score

CONTINUOUS = [
    "Motor current(mA)",
    "Motor Voltage(10mV)",
    "Motor electrodynamic force",
    "Door leaf position",
]


def parse_door_time(values: pd.Series) -> pd.Series:
    parts = values.astype(str).str.split("-", expand=True).astype(int)
    base = pd.to_datetime(
        {
            "year": parts[0], "month": parts[1], "day": parts[2],
            "hour": parts[3], "minute": parts[4], "second": parts[5],
        }
    )
    return base + pd.to_timedelta(parts[6], unit="ms")


def segment_stream(frame: pd.DataFrame, gap_factor: float = 10.0) -> list[tuple[int, int]]:
    times = parse_door_time(frame["Datetime"])
    delta_ms = times.diff().dt.total_seconds().mul(1000.0)
    normal_step = float(delta_ms[delta_ms > 0].median())
    boundaries = np.flatnonzero(delta_ms.to_numpy() > normal_step * gap_factor)
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries - 1, len(frame) - 1]
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _series_features(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    diff = np.diff(values)
    quantiles = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    result = {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_rms": float(np.sqrt(np.mean(values * values))),
        f"{prefix}_q05": float(quantiles[0]),
        f"{prefix}_q25": float(quantiles[1]),
        f"{prefix}_median": float(quantiles[2]),
        f"{prefix}_q75": float(quantiles[3]),
        f"{prefix}_q95": float(quantiles[4]),
        f"{prefix}_iqr": float(quantiles[3] - quantiles[1]),
        f"{prefix}_diff_mean": float(np.mean(diff)) if len(diff) else 0.0,
        f"{prefix}_diff_std": float(np.std(diff)) if len(diff) else 0.0,
        f"{prefix}_diff_maxabs": float(np.max(np.abs(diff))) if len(diff) else 0.0,
    }
    grid = np.linspace(0, max(len(values) - 1, 0), 16)
    sampled = np.interp(grid, np.arange(len(values)), values)
    result.update({f"{prefix}_sample_{idx:02d}": float(value) for idx, value in enumerate(sampled)})
    return result


def extract_segments(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for segment_number, (start, end) in enumerate(segment_stream(frame), start=1):
        segment = frame.iloc[start : end + 1]
        row: dict[str, float | str] = {
            "segment_number": segment_number,
            "start_time": str(segment.iloc[0]["Datetime"]),
            "end_time": str(segment.iloc[-1]["Datetime"]),
            "n_rows": len(segment),
            "operation_open": float(segment["Door is opening"].mean() >= segment["Door is closing"].mean()),
        }
        for column in CONTINUOUS:
            row.update(_series_features(segment[column].to_numpy(dtype=float), column.replace(" ", "_").lower()))
        rows.append(row)
    return pd.DataFrame(rows)


def _make_model(quick: bool = False) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=160 if quick else 600,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )


def train(data_dir: Path, model_dir: Path, quick: bool = False) -> dict:
    raw = pd.read_csv(data_dir / "Train.csv")
    truth = pd.read_csv(data_dir / "Train_Segments_Answer.csv")
    features = extract_segments(raw)
    labels = truth.set_index("start_time")["status"]
    features["target"] = features["start_time"].map(labels)
    if features["target"].isna().any():
        raise ValueError("Door timestamp-gap segments did not align with the supplied labels")

    feature_names = [column for column in features.columns if column not in {"segment_number", "start_time", "end_time", "target"}]
    X = finite_frame(features, feature_names)
    y = (features["target"] == "Abnormal resistance").astype(int).to_numpy()
    operation = features["operation_open"].astype(int).to_numpy()
    strata = np.asarray([f"{op}_{target}" for op, target in zip(operation, y)])
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    probabilities = cross_val_predict(_make_model(quick), X, y, cv=folds.split(X, strata), method="predict_proba", n_jobs=1)[:, 1]

    candidates = np.linspace(0.15, 0.85, 141)
    scores = [door_exact_boundary_score(y, probabilities >= threshold) for threshold in candidates]
    threshold = float(candidates[int(np.argmax(scores))])
    model = _make_model(quick)
    model.fit(X, y)

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": feature_names, "threshold": threshold}, model_dir / "door.joblib")
    report = {
        "segments": int(len(features)),
        "normal": int((y == 0).sum()),
        "abnormal": int((y == 1).sum()),
        "oof_official_score": float(max(scores)),
        "threshold": threshold,
    }
    write_json(model_dir / "door_metrics.json", report)
    return report


def predict(data_dir: Path, model_dir: Path, output_path: Path) -> pd.DataFrame:
    artifact = joblib.load(model_dir / "door.joblib")
    raw = pd.read_csv(data_dir / "Test.csv")
    features = extract_segments(raw)
    X = finite_frame(features, artifact["features"])
    probabilities = artifact["model"].predict_proba(X)[:, 1]
    labels = np.where(probabilities >= artifact["threshold"], "Abnormal resistance", "Normal")
    result = features[["start_time", "end_time"]].copy()
    result["prediction"] = labels
    validate_submission("door", result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result

