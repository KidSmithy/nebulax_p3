from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import LeaveOneGroupOut

from .common import SEED, finite_frame, validate_submission, write_json
from .metrics import acv_rank_score


def _canonical_parameter(name: str) -> str:
    text = name.lower().strip()
    replacements = {
        "outside temperature sensor reading": "outdoor temperature",
        "outdoor average temperature": "outdoor temperature",
        "indoor average temperature": "indoor temperature",
        "passenger cabin temperature detected value": "indoor temperature",
        "fresh air temperature detected value": "outdoor temperature",
        "acv control temperature (cooling)": "cooling control temperature",
        "acv control temperature (heating)": "heating control temperature",
        "load shedding": "load halved",
    }
    return replacements.get(text, text)


def _column_map(frame: pd.DataFrame) -> tuple[list[str], dict[str, dict[str, str]]]:
    pattern = re.compile(r"^Car (\d+) - (.+)$")
    mapping: dict[str, dict[str, str]] = {}
    for column in map(str, frame.columns):
        match = pattern.match(column)
        if match:
            car, parameter = match.groups()
            mapping.setdefault(car, {})[_canonical_parameter(parameter)] = column
    cars = sorted(mapping)
    if not cars:
        raise ValueError("No 'Car <NN> - <parameter>' columns found")
    return cars, mapping


def extract_case(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path)
    cars, mapping = _column_map(frame)
    parameters = sorted(set.intersection(*(set(mapping[car]) for car in cars)))
    rows = {car: {"file_id": path.name, "car": car} for car in cars}
    anomaly_components = {car: [] for car in cars}

    for parameter in parameters:
        matrix = np.column_stack(
            [pd.to_numeric(frame[mapping[car][parameter]], errors="coerce").to_numpy(dtype=float) for car in cars]
        )
        valid_rows = np.mean(np.isfinite(matrix), axis=1) >= 0.75
        if valid_rows.sum() < 20:
            continue
        matrix = matrix[valid_rows]
        row_median = np.nanmedian(matrix, axis=1, keepdims=True)
        row_mad = np.nanmedian(np.abs(matrix - row_median), axis=1, keepdims=True)
        fallback = np.nanstd(matrix, axis=1, keepdims=True)
        scale = np.where(row_mad > 1e-6, row_mad, fallback) + 1e-6
        zscore = (matrix - row_median) / scale

        key = re.sub(r"[^a-z0-9]+", "_", parameter).strip("_")
        for index, car in enumerate(cars):
            values = matrix[:, index]
            finite = values[np.isfinite(values)]
            zvalues = zscore[:, index]
            zfinite = zvalues[np.isfinite(zvalues)]
            if not len(finite) or not len(zfinite):
                continue
            rows[car].update(
                {
                    f"{key}_mean": float(np.mean(finite)),
                    f"{key}_std": float(np.std(finite)),
                    f"{key}_q05": float(np.quantile(finite, 0.05)),
                    f"{key}_q95": float(np.quantile(finite, 0.95)),
                    f"{key}_nunique": float(min(pd.Series(finite).nunique(), 1000)),
                    f"{key}_peer_abs_median": float(np.median(np.abs(zfinite))),
                    f"{key}_peer_abs_q95": float(np.quantile(np.abs(zfinite), 0.95)),
                    f"{key}_peer_signed_mean": float(np.mean(zfinite)),
                    f"{key}_peer_outlier_rate": float(np.mean(np.abs(zfinite) > 2.0)),
                }
            )
            anomaly_components[car].append(
                float(np.median(np.abs(zfinite)) + 0.25 * np.quantile(np.abs(zfinite), 0.95))
            )

    result = pd.DataFrame(rows.values())
    result["unsupervised_score"] = [float(np.mean(anomaly_components[car])) if anomaly_components[car] else 0.0 for car in cars]
    for temperature in ("indoor_temperature", "outdoor_temperature", "cooling_control_temperature"):
        mean_column = f"{temperature}_mean"
        if mean_column not in result:
            result[mean_column] = 0.0
    result["indoor_minus_outdoor"] = result["indoor_temperature_mean"] - result["outdoor_temperature_mean"]
    result["indoor_minus_control"] = result["indoor_temperature_mean"] - result["cooling_control_temperature_mean"]
    return result


def _make_model(quick: bool = False) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=200 if quick else 800,
        min_samples_leaf=2,
        max_features=0.7,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )


def train(data_dir: Path, model_dir: Path, quick: bool = False) -> dict:
    labels = pd.read_csv(data_dir / "Train_Labels.csv", dtype={"faulty_car": str})
    labels["faulty_car"] = labels["faulty_car"].str.zfill(2)
    all_features = pd.concat([extract_case(data_dir / "Train" / filename) for filename in labels["filename"]], ignore_index=True)
    all_features = all_features.merge(labels, left_on="file_id", right_on="filename", how="left")
    all_features["target"] = (all_features["car"] == all_features["faulty_car"]).astype(int)
    ignored = {"file_id", "car", "filename", "faulty_car", "target"}
    feature_names = sorted(column for column in all_features.columns if column not in ignored)
    X = finite_frame(all_features, feature_names)
    y = all_features["target"].to_numpy()
    groups = all_features["file_id"].to_numpy()

    supervised_oof = np.zeros(len(all_features), dtype=float)
    splitter = LeaveOneGroupOut()
    for train_index, valid_index in splitter.split(X, y, groups):
        model = _make_model(quick)
        model.fit(X.iloc[train_index], y[train_index])
        supervised_oof[valid_index] = model.predict_proba(X.iloc[valid_index])[:, 1]

    raw_anomaly = all_features["unsupervised_score"].to_numpy(dtype=float)
    normalized_anomaly = np.zeros_like(raw_anomaly)
    for group in np.unique(groups):
        mask = groups == group
        values = raw_anomaly[mask]
        normalized_anomaly[mask] = (values - values.mean()) / (values.std() + 1e-9)

    best_weight, best_score = 0.0, -1.0
    for weight in np.linspace(0.0, 1.0, 21):
        blend = weight * supervised_oof + (1.0 - weight) * normalized_anomaly
        scores = []
        for group in np.unique(groups):
            mask = groups == group
            ranking = all_features.loc[mask].iloc[np.argsort(-blend[mask])]["car"].tolist()
            truth = str(all_features.loc[mask, "faulty_car"].iloc[0])
            scores.append(acv_rank_score(truth, ranking))
        score = float(np.mean(scores))
        if score > best_score:
            best_weight, best_score = float(weight), score

    model = _make_model(quick)
    model.fit(X, y)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "features": feature_names, "supervised_weight": best_weight},
        model_dir / "acv.joblib",
    )
    report = {"cases": int(labels.shape[0]), "loco_rank_score": best_score, "supervised_weight": best_weight}
    write_json(model_dir / "acv_metrics.json", report)
    return report


def predict(data_dir: Path, model_dir: Path, output_path: Path) -> pd.DataFrame:
    artifact = joblib.load(model_dir / "acv.joblib")
    input_files = sorted((data_dir / "Test").glob("*.xlsx"))
    output_rows = []
    for path in input_files:
        features = extract_case(path)
        X = finite_frame(features, artifact["features"])
        supervised = artifact["model"].predict_proba(X)[:, 1]
        anomaly = features["unsupervised_score"].to_numpy(dtype=float)
        anomaly = (anomaly - anomaly.mean()) / (anomaly.std() + 1e-9)
        weight = artifact["supervised_weight"]
        score = weight * supervised + (1.0 - weight) * anomaly
        ranking = features.iloc[np.argsort(-score)]["car"].tolist()
        output_rows.append({"file_id": path.name, "ranked_cars": "|".join(ranking)})
    result = pd.DataFrame(output_rows, columns=["file_id", "ranked_cars"])
    validate_submission("acv", result, input_files)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result

