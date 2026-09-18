from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

SEED = 42
SUBSYSTEMS = ("door", "acv", "rail", "shm")


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def resolve_data_root(path: str | Path) -> Path:
    """Accept the PS3 folder, 02_Datasets folder, or repository root."""
    root = Path(path).expanduser().resolve()
    candidates = [
        root,
        root / "02_Datasets",
        root / "PS3" / "02_Datasets",
        root / "NebulaX-Hackathon-ProblemStatement" / "PS3" / "02_Datasets",
    ]
    for candidate in candidates:
        if all((candidate / name).exists() for name in ("Door", "ACV", "Rail_Corrugation", "SHM")):
            return candidate
    raise FileNotFoundError(
        f"Could not find Door, ACV, Rail_Corrugation, and SHM under {root}. "
        "Pass the PS3/02_Datasets directory with --data-root."
    )


def ensure_dirs(*paths: str | Path) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def finite_frame(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = frame.loc[:, list(columns)].copy()
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def write_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def describe_scores(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def validate_submission(name: str, frame: pd.DataFrame, input_files: list[Path] | None = None) -> None:
    if frame.empty:
        raise ValueError(f"{name} submission is empty")
    if frame.isna().any().any():
        raise ValueError(f"{name} submission contains missing values")

    if name == "door":
        required = ["start_time", "end_time", "prediction"]
        if list(frame.columns) != required:
            raise ValueError(f"Door columns must be exactly {required}")
        allowed = {"Normal", "Abnormal resistance"}
        if not set(frame["prediction"]).issubset(allowed):
            raise ValueError("Door prediction contains an invalid label")
    elif name == "acv":
        required = ["file_id", "ranked_cars"]
        if list(frame.columns) != required:
            raise ValueError(f"ACV columns must be exactly {required}")
        for ranking in frame["ranked_cars"]:
            cars = str(ranking).split("|")
            if len(cars) != len(set(cars)) or not cars:
                raise ValueError(f"Invalid ACV ranking: {ranking}")
    elif name == "rail":
        required = ["file_id", "prediction"]
        if list(frame.columns) != required:
            raise ValueError(f"Rail columns must be exactly {required}")
        if not set(frame["prediction"]).issubset({"Normal", "Side I", "Side II"}):
            raise ValueError("Rail prediction contains an invalid label")
    elif name == "shm":
        required = ["file_id", "prediction"]
        if list(frame.columns) != required:
            raise ValueError(f"SHM columns must be exactly {required}")
        values = pd.to_numeric(frame["prediction"], errors="coerce")
        if values.isna().any() or (values <= 0).any():
            raise ValueError("SHM predictions must be finite positive numbers")
    else:
        raise ValueError(f"Unknown subsystem: {name}")

    if input_files is not None and name != "door":
        expected = {path.name for path in input_files}
        actual = set(frame["file_id"].astype(str))
        if expected != actual:
            raise ValueError(f"{name} file IDs differ from input files: missing={expected-actual}, extra={actual-expected}")

