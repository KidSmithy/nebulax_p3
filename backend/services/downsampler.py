"""
backend/services/downsampler.py
===============================================================================
Largest-Triangle-Three-Buckets (LTTB) waveform downsampler.
Preserves critical visual peaks and dynamic range for high-frequency waveforms
(e.g., 1000 Hz vibration, 100 Hz door motor current) before streaming over WebSockets.
===============================================================================
"""

import numpy as np
from typing import List, Tuple, Union

def lttb_downsample(data: Union[np.ndarray, List[float]], threshold: int = 100) -> List[float]:
    """
    Downsamples a 1D sequence of values to `threshold` points using the
    Largest-Triangle-Three-Buckets (LTTB) algorithm.
    """
    if len(data) <= threshold or threshold <= 2:
        return [float(v) for v in data]

    # Convert to 2D coordinates: (index, value)
    n = len(data)
    y_vals = np.asarray(data, dtype=np.float64)
    x_vals = np.arange(n, dtype=np.float64)

    sampled = []
    # Bucket size. Leave room for start and end points
    every = (n - 2) / (threshold - 2)

    a = 0  # Initial point
    sampled.append(float(y_vals[a]))

    for i in range(threshold - 2):
        # Calculate point average for next bucket (bucket C)
        avg_x = 0.0
        avg_y = 0.0
        avg_range_start = int(np.floor((i + 1) * every) + 1)
        avg_range_end = int(np.floor((i + 2) * every) + 1)
        avg_range_end = min(avg_range_end, n)

        avg_range_len = avg_range_end - avg_range_start
        if avg_range_len > 0:
            avg_x = np.mean(x_vals[avg_range_start:avg_range_end])
            avg_y = np.mean(y_vals[avg_range_start:avg_range_end])
        else:
            avg_x = x_vals[-1]
            avg_y = y_vals[-1]

        # Get range for current bucket (bucket B)
        range_offs = int(np.floor(i * every) + 1)
        range_to = int(np.floor((i + 1) * every) + 1)
        range_to = min(range_to, n)

        # Point a
        point_a_x = x_vals[a]
        point_a_y = y_vals[a]

        max_area = -1.0
        next_a = range_offs

        for current_idx in range(range_offs, range_to):
            # Calculate triangle area over points: (point_a, current_point, avg_point)
            # Area = 0.5 * |(Ax - Cx)(By - Ay) - (Ax - Bx)(Cy - Ay)|
            area = abs(
                (point_a_x - avg_x) * (y_vals[current_idx] - point_a_y)
                - (point_a_x - x_vals[current_idx]) * (avg_y - point_a_y)
            ) * 0.5

            if area > max_area:
                max_area = area
                next_a = current_idx

        sampled.append(float(y_vals[next_a]))
        a = next_a

    # Always add the last point
    sampled.append(float(y_vals[-1]))
    return sampled
