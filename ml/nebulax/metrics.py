from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def mape(y_true, y_pred) -> float:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(truth - pred) / np.maximum(np.abs(truth), 1e-12)))


def shm_score(y_true, y_pred) -> float:
    return max(0.0, 1.0 - mape(y_true, y_pred))


def acv_rank_score(true_car: str, ranked_cars: list[str]) -> float:
    if true_car not in ranked_cars or not ranked_cars:
        return 0.0
    rank = ranked_cars.index(true_car) + 1
    return float((len(ranked_cars) - (rank - 1)) / len(ranked_cars))


def door_exact_boundary_score(y_true, y_pred) -> float:
    """With exact one-to-one boundaries, the official IoU-weighted F1 equals accuracy."""
    truth = np.asarray(y_true)
    pred = np.asarray(y_pred)
    return float(np.mean(truth == pred))

