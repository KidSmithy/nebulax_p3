"""
acv2.physics
===========================================================================
The physical model behind the diagnosis.

Why a refrigerant leak is visible at all
----------------------------------------
A car's saloon is a lumped thermal capacitance driven by ambient conduction,
solar and passenger gains, and cooled by its own ACV pack:

    C dT_in/dt = UA (T_out - T_in) + Q_gain - Q_evap(t)                    (1)

The evaporator duty is bounded by the refrigeration capacity of the pack,

    Q_evap(t) <= Q_max = m_dot * (h_out - h_in)                            (2)

and the refrigerant mass flow m_dot is set by the compressor displacement and
the *suction density*, which collapses when the circuit loses charge. So an
undercharged pack has a smaller Q_max. Setting dT_in/dt = 0 in (1):

    T_in* = T_out + (Q_gain - Q_max) / UA                                  (3)

Three consequences drive every feature in :mod:`acv2.features`:

1. **Capacity deficit.** If Q_max still exceeds the load, the controller
   simply modulates and the leak is invisible. Once the load exceeds Q_max the
   car settles above set point: T_in - T_set > 0 and growing.
2. **Load dependence.** From (3), the shortfall grows with (T_out - T_set).
   A *miscalibrated sensor* produces a constant offset with zero slope; a
   *capacity loss* produces a slope. This is what separates the two.
3. **Progression.** A leak is a monotonic process: the deficit grows over the
   days of the record, whereas a thermostat set differently or a heavier
   passenger load does not trend.

Since the eight cars of one consist share ambient, solar and (largely) service
pattern, the peer consist acts as a free reference model: the *relative*
elevation of one car against the median of its siblings cancels UA, Q_gain and
the weather, leaving the capacity term. That is the whole reason localisation
is possible from cabin temperature alone.

On the richer schema the leak is observed directly instead of inferred: each
pack has two independent refrigeration circuits, so the sibling circuit inside
the same car is an even better reference than a sibling car - it shares the
ambient, the cabin and the demand command exactly. An undercharged circuit
shows a depressed condensing pressure, a reduced pressure lift
(p_high - p_low, proportional to the work per unit mass) and intermittent
suction-pressure collapse as the evaporator starves.
===========================================================================
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg

EPS = 1e-9


# --------------------------------------------------------------------------
# robust statistics
# --------------------------------------------------------------------------
def nan_median(values) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def mad_scale(values) -> float:
    """Median absolute deviation rescaled to a Gaussian sigma."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    return float(1.4826 * np.median(np.abs(arr - np.median(arr))))


def robust_z(series: pd.Series, clip: float | None = None) -> pd.Series:
    """Peer-relative robust z-score, immune to the single faulty car itself."""
    clip = cfg.PHYSICS["z_clip"] if clip is None else clip
    vals = pd.to_numeric(series, errors="coerce")
    centre = nan_median(vals)
    scale = mad_scale(vals)
    if not np.isfinite(scale) or scale < EPS:
        # Degenerate spread (e.g. a quantised channel where most cars are
        # identical): fall back to the standard deviation, then to a flat zero.
        scale = float(vals.std(ddof=0))
        if not np.isfinite(scale) or scale < EPS:
            return pd.Series(0.0, index=vals.index)
    z = (vals - centre) / scale
    return z.clip(-clip, clip).fillna(0.0)


def peer_reference(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Leave-one-out median across cars, row by row.

    Using the median *of the other cars* rather than of all cars stops a badly
    leaking car from contaminating its own reference, which matters when only
    three or four cars are instrumented.
    """
    values = frame.to_numpy(dtype=float)
    n_cars = values.shape[1]
    out = np.full_like(values, np.nan)
    for j in range(n_cars):
        others = np.delete(values, j, axis=1)
        with np.errstate(all="ignore"):
            out[:, j] = np.nanmedian(others, axis=1)
    return pd.DataFrame(out, index=frame.index, columns=frame.columns)


# --------------------------------------------------------------------------
# conditioning on the operating point
# --------------------------------------------------------------------------
def stratified_mean(values: pd.DataFrame, strata: pd.Series,
                    min_rows: int | None = None) -> pd.Series:
    """
    Per-car mean of ``values``, averaged with equal weight over the operating
    strata (load bins, demand tiers, ...) that every car actually visits.

    This is the defence against the strongest confound in the dataset: cars do
    not experience identical duty histories, and a car that simply spent more
    of the record in hot weather would otherwise look faulty.
    """
    min_rows = cfg.PHYSICS["min_bin_rows"] if min_rows is None else min_rows
    strata = strata.reindex(values.index)
    per_stratum = []
    for key, rows in values.groupby(strata, dropna=True).groups.items():
        block = values.loc[rows]
        counts = block.notna().sum()
        if (counts < min_rows).any():
            continue
        per_stratum.append(block.mean())
    if not per_stratum:
        return values.mean()
    return pd.concat(per_stratum, axis=1).mean(axis=1)


def load_bins(load: pd.Series, n_bins: int | None = None) -> pd.Series:
    """Quantile bins of the thermal-load proxy; NaN where the load is unknown."""
    n_bins = cfg.PHYSICS["n_load_bins"] if n_bins is None else n_bins
    clean = pd.to_numeric(load, errors="coerce")
    if clean.notna().sum() < n_bins * cfg.PHYSICS["min_bin_rows"]:
        return pd.Series(np.nan, index=load.index)
    try:
        return pd.qcut(clean, n_bins, labels=False, duplicates="drop")
    except ValueError:
        return pd.Series(np.nan, index=load.index)


# --------------------------------------------------------------------------
# regression / trend estimators
# --------------------------------------------------------------------------
def ols_slope(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    """Least-squares slope dy/dx, NaN-safe."""
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    ok = np.isfinite(xv) & np.isfinite(yv)
    if ok.sum() < 3:
        return float("nan")
    xv, yv = xv[ok], yv[ok]
    var = float(np.var(xv))
    if var < EPS:
        return float("nan")
    return float(np.cov(xv, yv, bias=True)[0, 1] / var)


def binned_slope(load: pd.Series, elevation: pd.Series, n_bins: int | None = None) -> float:
    """
    Slope of elevation against thermal load, estimated on bin means.

    Binning first makes the estimate insensitive to the heavy quantisation of
    the temperature channels (0.25-0.5 K) and to the very uneven density of
    operating points, and it is far more robust than raw OLS on 9000 rows of
    correlated samples.
    """
    n_bins = cfg.PHYSICS["n_load_bins"] if n_bins is None else n_bins
    df = pd.DataFrame({"load": pd.to_numeric(load, errors="coerce"),
                       "elev": pd.to_numeric(elevation, errors="coerce")}).dropna()
    if len(df) < n_bins * cfg.PHYSICS["min_bin_rows"]:
        return float("nan")
    try:
        df["bin"] = pd.qcut(df["load"], n_bins, labels=False, duplicates="drop")
    except ValueError:
        return float("nan")
    grouped = df.groupby("bin").agg(load=("load", "mean"), elev=("elev", "mean"),
                                    n=("elev", "size"))
    grouped = grouped[grouped["n"] >= cfg.PHYSICS["min_bin_rows"]]
    if len(grouped) < 3:
        return float("nan")
    return ols_slope(grouped["load"], grouped["elev"])


def daily_trend(days: pd.Series, elevation: pd.Series) -> float:
    """
    Trend of the peer-relative elevation in K/day, estimated on daily means so
    that a diurnal cycle or an uneven sampling density cannot masquerade as a
    trend.
    """
    df = pd.DataFrame({"d": pd.to_numeric(days, errors="coerce"),
                       "e": pd.to_numeric(elevation, errors="coerce")}).dropna()
    if df.empty:
        return float("nan")
    span = df["d"].max() - df["d"].min()
    if span < 0.75:                       # less than ~18 h: no trend to speak of
        return float("nan")
    df["slot"] = np.floor(df["d"] * 4.0) / 4.0        # 6-hour slots
    grouped = df.groupby("slot").agg(d=("d", "mean"), e=("e", "mean"), n=("e", "size"))
    grouped = grouped[grouped["n"] >= cfg.PHYSICS["min_bin_rows"]]
    if len(grouped) < 4:
        return float("nan")
    return ols_slope(grouped["d"], grouped["e"])


# --------------------------------------------------------------------------
# grey-box identification
# --------------------------------------------------------------------------
def identify_thermal_model(t_in: pd.Series, ambient: pd.Series, cooling: pd.Series,
                           segment: pd.Series, dt_seconds: float,
                           smooth_rows: int = 5) -> dict:
    """
    Fit the lumped model (1) in the discrete form

        dT_in/dt = a (T_out - T_in) + b * u + c ,    u = 1 while cooling

    by ordinary least squares, where ``a = UA/C`` [1/h] is the envelope
    coupling and ``-b`` [K/h] is the achievable cooling rate of the pack, i.e.
    a direct estimate of ``Q_max / C``. Derivatives never cross a data gap.

    Returns the identified parameters plus the fit quality, so a car whose
    model did not identify (too little mode variation) can be excluded rather
    than silently contributing a meaningless number.
    """
    t = pd.to_numeric(t_in, errors="coerce")
    smooth = t.groupby(segment).transform(
        lambda s: s.rolling(smooth_rows, min_periods=2, center=True).mean())
    dtdt = smooth.groupby(segment).diff() * (3600.0 / max(dt_seconds, EPS))   # K/h
    drive = pd.to_numeric(ambient, errors="coerce") - smooth
    u = cooling.astype(float)

    df = pd.DataFrame({"y": dtdt, "x1": drive, "x2": u}).dropna()
    # A ramp identification needs both cooling and non-cooling samples.
    if len(df) < 50 or df["x2"].nunique() < 2 or df["x2"].std() < 0.02:
        return {"ua_over_c": float("nan"), "cool_rate": float("nan"),
                "bias": float("nan"), "r2": float("nan"), "n": int(len(df))}
    design = np.column_stack([df["x1"].to_numpy(), df["x2"].to_numpy(),
                              np.ones(len(df))])
    target = df["y"].to_numpy()
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    pred = design @ coef
    ss_res = float(np.sum((target - pred) ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    return {
        "ua_over_c": float(coef[0]),
        "cool_rate": float(-coef[1]),       # K/h of cooling authority
        "bias": float(coef[2]),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > EPS else float("nan"),
        "n": int(len(df)),
    }


def pulldown_rate(t_in: pd.Series, cooling: pd.Series, segment: pd.Series,
                  dt_seconds: float, smooth_minutes: float | None = None,
                  min_rows: int | None = None) -> float:
    """
    Peak sustained pull-down rate in K/h (negative), measured on cooling ramps.

    A pack that has lost charge cannot pull the cabin down as fast, regardless
    of where the controller eventually settles, so this probes Q_max directly
    and is independent of the set point the car happens to be given.
    """
    smooth_minutes = cfg.PHYSICS["pulldown_smooth_min"] if smooth_minutes is None else smooth_minutes
    min_rows = cfg.PHYSICS["pulldown_min_rows"] if min_rows is None else min_rows
    rows = max(2, int(round(smooth_minutes * 60.0 / max(dt_seconds, EPS))))
    t = pd.to_numeric(t_in, errors="coerce")
    smooth = t.groupby(segment).transform(
        lambda s: s.rolling(rows, min_periods=2, center=True).mean())
    rate = smooth.groupby(segment).diff() * (3600.0 / max(dt_seconds, EPS))
    rate = rate.where(cooling.astype(bool))
    falling = rate < 0
    # Keep only runs of consecutive falling samples: isolated negative steps are
    # quantisation noise, not a pull-down.
    run_id = (falling != falling.shift()).cumsum()
    run_len = falling.groupby(run_id).transform("size")
    sustained = rate.where(falling & (run_len >= min_rows))
    valid = sustained.dropna()
    if len(valid) < min_rows:
        return float("nan")
    return float(np.percentile(valid, 5))          # most negative sustained rate


# --------------------------------------------------------------------------
# change detection
# --------------------------------------------------------------------------
def cusum_onset(residual: pd.Series, days: pd.Series,
                k: float | None = None, h: float | None = None,
                clip: float | None = None) -> dict:
    """
    One-sided CUSUM on the peer-relative residual, used as a leak-onset
    detector: ``S_i = max(0, S_{i-1} + clip(r_i) - k)``, alarm when ``S_i > h``.

    ``k`` is the slack the residual is allowed without being called drift, so
    normal inter-car scatter accumulates nothing. Because the statistic is an
    integral, a small but persistent elevation - the signature of a slow leak -
    eventually alarms. Each step is capped at ``clip`` kelvin so that a single
    sensor glitch cannot trip the alarm on its own: the detector responds to
    *persistence*, which is what distinguishes a leak from an artefact.
    """
    k = cfg.PHYSICS["cusum_k_c"] if k is None else k
    h = cfg.PHYSICS["cusum_h_c"] if h is None else h
    clip = cfg.PHYSICS["cusum_clip_c"] if clip is None else clip
    df = pd.DataFrame({"r": pd.to_numeric(residual, errors="coerce"),
                       "d": pd.to_numeric(days, errors="coerce")}).dropna()
    if df.empty:
        return {"detected": False, "onset_day": float("nan"),
                "fraction_after": 0.0, "peak": 0.0}
    # Work on 15-minute means so that the statistic has the same physical
    # meaning at 10 s and 30 s sampling.
    df["slot"] = np.floor(df["d"] * 96.0)
    series = df.groupby("slot").agg(r=("r", "mean"), d=("d", "mean"))
    s = 0.0
    peak = 0.0
    onset_day = float("nan")
    for day, resid in zip(series["d"].to_numpy(),
                          np.clip(series["r"].to_numpy(), -clip, clip)):
        s = max(0.0, s + resid - k)
        peak = max(peak, s)
        if s > h and not np.isfinite(onset_day):
            onset_day = float(day)
    total_span = float(series["d"].max() - series["d"].min())
    if np.isfinite(onset_day) and total_span > EPS:
        fraction_after = float((series["d"].max() - onset_day) / total_span)
    else:
        fraction_after = 0.0
    return {"detected": bool(np.isfinite(onset_day)), "onset_day": onset_day,
            "fraction_after": max(0.0, min(1.0, fraction_after)), "peak": peak}


# --------------------------------------------------------------------------
# refrigerant-circuit thermodynamics
# --------------------------------------------------------------------------
def circuit_indicators(p_low: pd.Series, p_high: pd.Series) -> dict:
    """
    Condition-free undercharge indicators for one refrigeration circuit.

    * ``lift = p_high - p_low`` is proportional to the compression work per
      unit mass and falls when the circuit is starved of refrigerant.
    * ``ratio = p_high / p_low`` rises, because the suction side collapses
      proportionally more than the discharge side.
    * ``suction_excursion = p50 - p05`` captures intermittent evaporator
      starvation: a correctly charged circuit holds its suction pressure in a
      narrow band, an undercharged one repeatedly dives toward the low-pressure
      cut-out.
    """
    lo = pd.to_numeric(p_low, errors="coerce").dropna()
    hi = pd.to_numeric(p_high, errors="coerce").dropna()
    common = lo.index.intersection(hi.index)
    lo_c, hi_c = lo.loc[common], hi.loc[common]
    if len(common) < 20:
        return {k: float("nan") for k in
                ("p_low_med", "p_high_med", "lift", "ratio",
                 "suction_excursion", "suction_sd", "n")} | {"n": int(len(common))}
    lift = hi_c - lo_c
    ratio = hi_c / lo_c.replace(0.0, np.nan)
    return {
        "p_low_med": float(lo_c.median()),
        "p_high_med": float(hi_c.median()),
        "lift": float(lift.median()),
        "ratio": float(ratio.median()),
        "suction_excursion": float(lo_c.median() - np.percentile(lo_c, 5)),
        "suction_sd": float(lo_c.std()),
        "n": int(len(common)),
    }
