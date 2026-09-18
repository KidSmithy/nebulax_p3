from __future__ import annotations

from collections import deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import SEED, finite_frame, validate_submission, write_json
from .metrics import mape, shm_score

RAINFLOW_POWERS = (3, 4, 5, 6, 8, 10, 12)


def reversals(series: np.ndarray) -> np.ndarray:
    values = np.asarray(series, dtype=float)
    if len(values) < 3:
        return values
    values = values[np.r_[True, values[1:] != values[:-1]]]
    if len(values) < 3:
        return values
    delta = np.diff(values)
    turning = np.r_[True, delta[:-1] * delta[1:] < 0, True]
    return values[turning]


def rainflow_cycles(series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points: deque[float] = deque()
    amplitudes: list[float] = []
    counts: list[float] = []
    for point in reversals(series):
        points.append(float(point))
        while len(points) >= 3:
            first = abs(points[-2] - points[-3])
            second = abs(points[-1] - points[-2])
            if first > second:
                break
            if len(points) == 3:
                amplitudes.append(first / 2.0)
                counts.append(0.5)
                points.popleft()
            else:
                amplitudes.append(first / 2.0)
                counts.append(1.0)
                last = points.pop()
                points.pop()
                points.pop()
                points.append(last)
    while len(points) > 1:
        first = points.popleft()
        second = points[0]
        amplitudes.append(abs(second - first) / 2.0)
        counts.append(0.5)
    return np.asarray(amplitudes, dtype=float), np.asarray(counts, dtype=float)


def _log_damage_index(amplitudes: np.ndarray, counts: np.ndarray, power: int) -> float:
    mask = (amplitudes > 0) & (counts > 0)
    if not np.any(mask):
        return -30.0
    terms = np.log(counts[mask]) + power * np.log(amplitudes[mask])
    maximum = float(np.max(terms))
    return maximum + float(np.log(np.sum(np.exp(terms - maximum))))


def extract_file(path: Path) -> dict[str, float | str]:
    values = pd.read_csv(path, header=None, dtype=np.float32).iloc[:, 0].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    absolute = np.abs(values)
    centered = values - np.mean(values)
    std = float(np.std(values)) + 1e-12
    diff = np.diff(values)
    row: dict[str, float | str] = {
        "file_id": path.name,
        "mean": float(np.mean(values)),
        "std": std,
        "rms": float(np.sqrt(np.mean(values * values))),
        "max_abs": float(np.max(absolute)),
        "q50_abs": float(np.quantile(absolute, 0.50)),
        "q75_abs": float(np.quantile(absolute, 0.75)),
        "q90_abs": float(np.quantile(absolute, 0.90)),
        "q95_abs": float(np.quantile(absolute, 0.95)),
        "q99_abs": float(np.quantile(absolute, 0.99)),
        "q995_abs": float(np.quantile(absolute, 0.995)),
        "q999_abs": float(np.quantile(absolute, 0.999)),
        "skew": float(np.mean((centered / std) ** 3)),
        "kurtosis": float(np.mean((centered / std) ** 4)),
        "crest_factor": float(np.max(absolute) / (np.sqrt(np.mean(values * values)) + 1e-12)),
        "diff_rms": float(np.sqrt(np.mean(diff * diff))),
        "turning_rate": float(np.mean(np.sign(diff[1:]) != np.sign(diff[:-1]))),
        "zero_crossing_rate": float(np.mean(np.signbit(values[1:]) != np.signbit(values[:-1]))),
    }
    for power in (1, 2, 3, 4, 5, 6, 8, 10, 12):
        row[f"power_mean_{power}"] = float(np.mean(absolute**power) ** (1.0 / power))

    chunks = np.array_split(values, 32)
    chunk_rms = np.array([np.sqrt(np.mean(chunk * chunk)) for chunk in chunks])
    row.update(
        {
            "chunk_rms_mean": float(np.mean(chunk_rms)),
            "chunk_rms_std": float(np.std(chunk_rms)),
            "chunk_rms_min": float(np.min(chunk_rms)),
            "chunk_rms_max": float(np.max(chunk_rms)),
        }
    )

    # Normalized spectral bands; sample rate is not disclosed, so bands are fractions of Nyquist.
    target_length = min(len(values), 131072)
    indices = np.linspace(0, len(values) - 1, target_length).astype(int)
    sampled = values[indices] - np.mean(values[indices])
    power_spectrum = np.abs(np.fft.rfft(sampled)) ** 2
    total = float(np.sum(power_spectrum[1:])) + 1e-12
    edges = np.linspace(0, len(power_spectrum), 17).astype(int)
    for index in range(16):
        row[f"spectral_band_{index:02d}"] = float(np.sum(power_spectrum[edges[index] : edges[index + 1]]) / total)

    amplitudes, counts = rainflow_cycles(values)
    row["rainflow_cycle_count"] = float(np.sum(counts))
    if len(amplitudes):
        row["rainflow_amp_mean"] = float(np.average(amplitudes, weights=counts))
        row["rainflow_amp_q95"] = float(np.quantile(amplitudes, 0.95))
        row["rainflow_amp_max"] = float(np.max(amplitudes))
    else:
        row["rainflow_amp_mean"] = row["rainflow_amp_q95"] = row["rainflow_amp_max"] = 0.0
    for power in RAINFLOW_POWERS:
        row[f"rainflow_log_damage_p{power}"] = _log_damage_index(amplitudes, counts, power)
    return row


def _signature(paths: list[Path]) -> list[tuple[str, int, int]]:
    return [(path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in paths]


def extract_many(paths: list[Path], cache_path: Path | None = None) -> pd.DataFrame:
    signature = _signature(paths)
    if cache_path and cache_path.exists():
        cached = joblib.load(cache_path)
        if cached.get("signature") == signature:
            return cached["features"]
    rows = []
    for index, path in enumerate(paths, start=1):
        print(f"[shm features] {index}/{len(paths)} {path.name}", flush=True)
        rows.append(extract_file(path))
    features = pd.DataFrame(rows)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"signature": signature, "features": features}, cache_path, compress=3)
    return features


def _ridge_model():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 4, 33)))


def _tree_model(quick: bool = False):
    return ExtraTreesRegressor(
        n_estimators=160 if quick else 700,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=SEED,
        n_jobs=-1,
    )


def train(data_dir: Path, model_dir: Path, quick: bool = False) -> dict:
    labels = pd.read_csv(data_dir / "Train_Labels.csv")
    paths = [data_dir / "Train" / filename for filename in labels["filename"]]
    features = extract_many(paths, model_dir / "cache" / "shm_train_features.joblib")
    table = labels.merge(features, left_on="filename", right_on="file_id", validate="one_to_one")
    feature_names = sorted(column for column in features.columns if column != "file_id")
    X = finite_frame(table, feature_names)
    y = table["damage"].to_numpy(dtype=float)
    log_y = np.log(np.maximum(y, 1e-12))
    ridge_oof = np.zeros(len(y))
    tree_oof = np.zeros(len(y))
    splitter = KFold(n_splits=8, shuffle=True, random_state=SEED)
    for fold, (train_index, valid_index) in enumerate(splitter.split(X), start=1):
        print(f"[shm cv] fold {fold}/8", flush=True)
        ridge = _ridge_model()
        tree = _tree_model(quick)
        ridge.fit(X.iloc[train_index], log_y[train_index])
        tree.fit(X.iloc[train_index], log_y[train_index])
        ridge_oof[valid_index] = np.exp(ridge.predict(X.iloc[valid_index]))
        tree_oof[valid_index] = np.exp(tree.predict(X.iloc[valid_index]))

    best_weight, best_mape = 0.0, float("inf")
    for weight in np.linspace(0.0, 1.0, 101):
        prediction = weight * ridge_oof + (1.0 - weight) * tree_oof
        score = mape(y, prediction)
        if score < best_mape:
            best_weight, best_mape = float(weight), score

    ridge = _ridge_model()
    tree = _tree_model(quick)
    ridge.fit(X, log_y)
    tree.fit(X, log_y)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"ridge": ridge, "tree": tree, "features": feature_names, "ridge_weight": best_weight},
        model_dir / "shm.joblib",
        compress=3,
    )
    report = {
        "files": int(len(table)),
        "oof_mape": best_mape,
        "oof_score": shm_score(y, best_weight * ridge_oof + (1.0 - best_weight) * tree_oof),
        "ridge_weight": best_weight,
    }
    write_json(model_dir / "shm_metrics.json", report)
    return report


def predict(data_dir: Path, model_dir: Path, output_path: Path) -> pd.DataFrame:
    artifact = joblib.load(model_dir / "shm.joblib")
    input_files = sorted((data_dir / "Test").glob("*.csv"))
    features = extract_many(input_files, model_dir / "cache" / "shm_test_features.joblib")
    X = finite_frame(features, artifact["features"])
    ridge_prediction = np.exp(artifact["ridge"].predict(X))
    tree_prediction = np.exp(artifact["tree"].predict(X))
    weight = artifact["ridge_weight"]
    prediction = np.maximum(weight * ridge_prediction + (1.0 - weight) * tree_prediction, 1e-12)
    result = pd.DataFrame({"file_id": features["file_id"], "prediction": prediction})
    validate_submission("shm", result, input_files)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result

