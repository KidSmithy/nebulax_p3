from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

from .common import SEED, finite_frame, validate_submission, write_json
from .metrics import macro_f1

LABELS = np.array(["Normal", "Side I", "Side II"])
FREQUENCY_BANDS = ((0, 50), (50, 100), (100, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 3500), (3500, 5001))


def _pool(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_q25": float(np.quantile(values, 0.25)),
        f"{prefix}_q75": float(np.quantile(values, 0.75)),
    }


def extract_file(path: Path) -> dict[str, float | str]:
    frame = pd.read_csv(path, dtype=np.float32)
    speed = frame.iloc[:, 0].to_numpy(dtype=np.float32)
    signals = frame.iloc[:, 1:].to_numpy(dtype=np.float32)
    columns = list(map(str, frame.columns[1:]))
    positions = np.array([int(re.search(r"position (\d+)", column).group(1)) for column in columns])
    cars = np.array([int(re.search(r"car (\d+)", column).group(1)) for column in columns])
    kinds = np.array(["vibration" if column.startswith("Vibration") else "shock" for column in columns])
    sides = np.where(positions % 2 == 1, "side_i", "side_ii")

    finite = np.isfinite(signals)
    if not finite.all():
        medians = np.nanmedian(np.where(finite, signals, np.nan), axis=0)
        signals = np.where(finite, signals, medians)
    means = np.mean(signals, axis=0)
    centered = signals - means
    std = np.std(signals, axis=0) + 1e-9
    rms = np.sqrt(np.mean(signals * signals, axis=0))
    max_abs = np.max(np.abs(signals), axis=0)
    q95_abs = np.quantile(np.abs(signals), 0.95, axis=0)
    kurtosis = np.mean((centered / std) ** 4, axis=0)
    crest = max_abs / (rms + 1e-9)

    spectrum = np.fft.rfft(centered, axis=0)
    power = np.abs(spectrum) ** 2
    total_power = np.sum(power[1:], axis=0) + 1e-12
    frequencies = np.fft.rfftfreq(len(signals), d=1.0 / 10000.0)
    centroid = np.sum(frequencies[:, None] * power, axis=0) / (np.sum(power, axis=0) + 1e-12)
    peak_frequency = frequencies[np.argmax(power[1:], axis=0) + 1]

    row: dict[str, float | str] = {
        "file_id": path.name,
        "speed_transition_count": float(np.count_nonzero(np.diff(speed))),
        "speed_transition_rate": float(np.mean(np.diff(speed) != 0)),
        "speed_active_fraction": float(np.mean(speed > np.nanmedian(speed))),
        "speed_mean": float(np.nanmean(speed)),
        "speed_std": float(np.nanstd(speed)),
    }
    channel_features = {
        "signal_mean": means,
        "signal_std": std,
        "rms": rms,
        "max_abs": max_abs,
        "q95_abs": q95_abs,
        "kurtosis": kurtosis,
        "crest": crest,
        "spectral_centroid": centroid,
        "peak_frequency": peak_frequency,
    }

    for side in ("side_i", "side_ii"):
        for kind in ("vibration", "shock"):
            mask = (sides == side) & (kinds == kind)
            prefix = f"{side}_{kind}"
            for feature_name, values in channel_features.items():
                row.update(_pool(values[mask], f"{prefix}_{feature_name}"))
            for low, high in FREQUENCY_BANDS:
                band_mask = (frequencies >= low) & (frequencies < high)
                relative_power = np.sum(power[band_mask][:, mask], axis=0) / total_power[mask]
                row.update(_pool(relative_power, f"{prefix}_band_{low}_{high}"))

    # Car-level pooling preserves distributed-versus-local fault information.
    for car in range(1, 9):
        for side in ("side_i", "side_ii"):
            mask = (cars == car) & (sides == side) & (kinds == "vibration")
            row[f"car_{car}_{side}_vibration_rms"] = float(np.mean(rms[mask]))
            row[f"car_{car}_{side}_vibration_q95"] = float(np.mean(q95_abs[mask]))

    for key in list(row):
        if "side_i" not in key:
            continue
        other = key.replace("side_i", "side_ii")
        if other not in row or not isinstance(row[key], (int, float)):
            continue
        left = float(row[key])
        right = float(row[other])
        suffix = key.replace("side_i", "side_contrast")
        row[f"{suffix}_difference"] = left - right
        row[f"{suffix}_relative"] = (left - right) / (abs(left) + abs(right) + 1e-9)
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
        print(f"[rail features] {index}/{len(paths)} {path.name}", flush=True)
        rows.append(extract_file(path))
    features = pd.DataFrame(rows)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"signature": signature, "features": features}, cache_path, compress=3)
    return features


def _swap_sides(frame: pd.DataFrame) -> pd.DataFrame:
    swapped = frame.copy()
    for column in frame.columns:
        if "side_i" in column:
            other = column.replace("side_i", "side_ii")
            if other in frame.columns:
                swapped[column] = frame[other].to_numpy()
        elif "side_ii" in column:
            other = column.replace("side_ii", "side_i")
            if other in frame.columns:
                swapped[column] = frame[other].to_numpy()
        elif "side_contrast" in column and column.endswith(("_difference", "_relative")):
            swapped[column] = -frame[column].to_numpy()
    return swapped


def _swap_labels(labels: np.ndarray) -> np.ndarray:
    return np.asarray(["Side II" if value == "Side I" else "Side I" if value == "Side II" else value for value in labels])


def _make_model(quick: bool = False):
    return ExtraTreesClassifier(
        n_estimators=140 if quick else 500,
        min_samples_leaf=2,
        max_features=0.7,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )


def _aligned_proba(model, X: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(X)
    aligned = np.zeros((len(X), len(LABELS)), dtype=float)
    for source, label in enumerate(model.classes_):
        aligned[:, int(np.flatnonzero(LABELS == label)[0])] = raw[:, source]
    return aligned


def train(data_dir: Path, model_dir: Path, quick: bool = False) -> dict:
    labels = pd.read_csv(data_dir / "Train_Labels.csv")
    paths = [data_dir / "Train" / filename for filename in labels["filename"]]
    features = extract_many(paths, model_dir / "cache" / "rail_train_features.joblib")
    table = labels.merge(features, left_on="filename", right_on="file_id", validate="one_to_one")
    feature_names = sorted(column for column in features.columns if column != "file_id")
    X = finite_frame(table, feature_names)
    y = table["label"].to_numpy()
    oof = np.zeros((len(X), len(LABELS)), dtype=float)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for fold, (train_index, valid_index) in enumerate(splitter.split(X, y), start=1):
        print(f"[rail cv] fold {fold}/5", flush=True)
        X_train = X.iloc[train_index]
        y_train = y[train_index]
        X_augmented = pd.concat([X_train, _swap_sides(X_train)], ignore_index=True)
        y_augmented = np.r_[y_train, _swap_labels(y_train)]
        extra = _make_model(quick)
        extra.fit(X_augmented, y_augmented)
        oof[valid_index] = _aligned_proba(extra, X.iloc[valid_index])

    best_multipliers = np.ones(3)
    best_score = -1.0
    for normal in (0.7, 0.85, 1.0, 1.15):
        for side_i in (0.9, 1.1, 1.3, 1.5, 1.8):
            for side_ii in (0.9, 1.1, 1.3, 1.5, 1.8):
                multipliers = np.array([normal, side_i, side_ii])
                prediction = LABELS[np.argmax(oof * multipliers, axis=1)]
                score = macro_f1(y, prediction)
                if score > best_score:
                    best_score = score
                    best_multipliers = multipliers

    final_prediction = LABELS[np.argmax(oof * best_multipliers, axis=1)]
    report_text = classification_report(y, final_prediction, labels=LABELS.tolist(), output_dict=True, zero_division=0)
    X_augmented = pd.concat([X, _swap_sides(X)], ignore_index=True)
    y_augmented = np.r_[y, _swap_labels(y)]
    extra = _make_model(quick)
    extra.fit(X_augmented, y_augmented)

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "extra": extra,
            "features": feature_names,
            "multipliers": best_multipliers,
        },
        model_dir / "rail.joblib",
        compress=3,
    )
    report = {
        "files": int(len(table)),
        "oof_macro_f1": best_score,
        "probability_multipliers": best_multipliers.tolist(),
        "per_class": {label: report_text[label] for label in LABELS},
    }
    write_json(model_dir / "rail_metrics.json", report)
    return report


def predict(data_dir: Path, model_dir: Path, output_path: Path) -> pd.DataFrame:
    artifact = joblib.load(model_dir / "rail.joblib")
    input_files = sorted((data_dir / "Test").glob("*.csv"), key=lambda path: int(re.search(r"\d+", path.stem).group()))
    features = extract_many(input_files, model_dir / "cache" / "rail_test_features.joblib")
    X = finite_frame(features, artifact["features"])
    probabilities = _aligned_proba(artifact["extra"], X)
    labels = LABELS[np.argmax(probabilities * artifact["multipliers"], axis=1)]
    result = pd.DataFrame({"file_id": features["file_id"], "prediction": labels})
    validate_submission("rail", result, input_files)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result
