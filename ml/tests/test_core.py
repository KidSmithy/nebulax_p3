from __future__ import annotations

import numpy as np
import pandas as pd

from nebulax.common import validate_submission
from nebulax.door import segment_stream
from nebulax.metrics import acv_rank_score, door_exact_boundary_score, shm_score
from nebulax.shm import rainflow_cycles


def test_door_gap_segmentation():
    frame = pd.DataFrame(
        {
            "Datetime": [
                "2023-7-5-0-0-0-0",
                "2023-7-5-0-0-0-20",
                "2023-7-5-0-0-10-0",
                "2023-7-5-0-0-10-20",
            ]
        }
    )
    assert segment_stream(frame) == [(0, 1), (2, 3)]


def test_metrics():
    assert door_exact_boundary_score([0, 1, 1], [0, 1, 0]) == 2 / 3
    assert acv_rank_score("03", ["01", "03", "02", "04"]) == 0.75
    assert shm_score([1.0, 2.0], [1.0, 2.0]) == 1.0


def test_rainflow_returns_positive_cycles():
    amplitudes, counts = rainflow_cycles(np.array([0.0, 2.0, 0.0, 2.0, 0.0]))
    assert len(amplitudes) == len(counts)
    assert np.all(amplitudes >= 0)
    assert np.sum(counts) > 0


def test_submission_validation():
    validate_submission(
        "door",
        pd.DataFrame(
            {
                "start_time": ["2023-1-1-0-0-0-0"],
                "end_time": ["2023-1-1-0-0-1-0"],
                "prediction": ["Normal"],
            }
        ),
    )

