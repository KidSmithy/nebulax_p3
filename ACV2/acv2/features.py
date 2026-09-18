"""
acv2.features
===========================================================================
Physics-informed feature extraction.

Every feature is a *per-car scalar* derived from the model in
:mod:`acv2.physics`, and every one of them is either
  (a) an estimate of the pack's cooling capacity or of the deficit it leaves
      behind (thermal / capacity groups),
  (b) evidence that the deficit is progressive rather than static
      (progression group),
  (c) a direct measurement on the refrigerant circuit (refrigerant group), or
  (d) the pack's own admission that something is wrong (integrity group).

Two branches are produced. The thermal branch works on any file, because every
schema in this dataset carries a cabin temperature, a set point and a running
mode. The refrigerant branch only activates on files that expose the high- and
low-side pressures of the two refrigeration circuits, and it is by far the more
direct evidence when it is available.
===========================================================================
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from . import physics as phys
from .cleaning import CleanCase

EPS = 1e-9


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _safe(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _strat_mean(values: pd.Series, strata: pd.Series, min_rows: int | None = None) -> float:
    """Mean of ``values`` averaged with equal weight over populated strata."""
    min_rows = cfg.PHYSICS["min_bin_rows"] if min_rows is None else min_rows
    df = pd.DataFrame({"v": pd.to_numeric(values, errors="coerce"), "s": strata}).dropna()
    if df.empty:
        return float("nan")
    grouped = df.groupby("s")["v"].agg(["mean", "size"])
    grouped = grouped[grouped["size"] >= min_rows]
    if grouped.empty:
        return float(df["v"].mean())
    return float(grouped["mean"].mean())


def _rate_per_hour(flag: pd.Series, dt_seconds: float) -> float:
    """Rising-edge rate of a boolean channel, per hour."""
    f = pd.to_numeric(flag, errors="coerce")
    valid = f.dropna()
    if len(valid) < 10:
        return float("nan")
    starts = int(((valid > 0.5) & (valid.shift(1) <= 0.5)).sum())
    hours = len(valid) * dt_seconds / 3600.0
    return starts / hours if hours > EPS else float("nan")


# --------------------------------------------------------------------------
# the operating context shared by every car
# --------------------------------------------------------------------------
def build_context(case: CleanCase) -> dict:
    """
    Peer-relative elevation, thermal-load proxy and load strata.

    The elevation is computed from cabin temperatures *masked to cooling rows
    only*, so a sibling that happens to be ventilating (and therefore warm)
    never pollutes the reference another car is judged against.
    """
    cars = case.cars
    t_in_cool = case.t_in[cars].where(case.cooling[cars])
    reference = phys.peer_reference(t_in_cool)
    elevation = t_in_cool - reference

    t_set_ref = case.t_set[cars].median(axis=1, skipna=True)
    load = case.ambient - t_set_ref           # the cooling span the packs must deliver
    bins = phys.load_bins(load)

    return {
        "cars": cars,
        "t_in_cool": t_in_cool,
        "reference": reference,
        "elevation": elevation,
        "load": load,
        "load_bins": bins,
        "days": case.elapsed_days,
    }


# --------------------------------------------------------------------------
# thermal branch
# --------------------------------------------------------------------------
def thermal_features(case: CleanCase, ctx: dict) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    window = case.panel.rows_for_minutes(cfg.PHYSICS["persist_window_min"])
    threshold = cfg.PHYSICS["elevation_threshold_c"]

    for car in ctx["cars"]:
        elev = ctx["elevation"][car]
        cooling = case.cooling[car].astype(bool)
        t_in = case.t_in[car]
        t_set = case.t_set[car]

        # --- static deficit ---------------------------------------------
        feats: dict[str, float] = {}
        feats["elev_mean"] = _safe(elev.mean())
        feats["elev_load_stratified"] = _strat_mean(elev, ctx["load_bins"])
        feats["elev_p90"] = _safe(elev.quantile(0.90))

        rolling = elev.groupby(case.segment).transform(
            lambda s: s.rolling(window, min_periods=max(3, window // 3)).mean())
        sustained = rolling.dropna()
        feats["elev_persistence"] = _safe((sustained > threshold).mean()) if len(sustained) else float("nan")

        err = (t_in - t_set).where(cooling)
        feats["setpoint_error"] = _safe(err.mean())

        # --- capacity: load dependence separates a leak from a bias ------
        feats["load_sensitivity"] = phys.binned_slope(ctx["load"].where(cooling), elev)

        span = (case.ambient - t_set).where(cooling)
        demanded = span.where(span > 3.0)
        shortfall = (err / demanded).replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
        # The mean, not the median: both temperature channels are quantised to
        # the same 0.5 K grid, so a pack that holds set point most of the time
        # has a median shortfall of exactly zero for every car. The mean keeps
        # the integral of the deficit, which is the physically meaningful part.
        feats["capacity_shortfall"] = _safe(shortfall.mean())

        feats["pulldown_rate"] = phys.pulldown_rate(
            t_in, cooling, case.segment, case.panel.dt_seconds)

        model = phys.identify_thermal_model(
            t_in, case.ambient, cooling, case.segment, case.panel.dt_seconds)
        feats["greybox_cool_rate"] = model["cool_rate"] if (
            np.isfinite(model["r2"]) and model["r2"] > 0.02) else float("nan")
        feats["_greybox_ua_over_c"] = model["ua_over_c"]
        feats["_greybox_r2"] = model["r2"]

        # --- progression -------------------------------------------------
        feats["elev_trend"] = phys.daily_trend(ctx["days"].where(cooling), elev)
        cusum = phys.cusum_onset(elev, ctx["days"])
        feats["cusum_fraction"] = cusum["fraction_after"]
        feats["_cusum_detected"] = float(cusum["detected"])
        feats["_cusum_onset_day"] = cusum["onset_day"]
        feats["_cusum_peak"] = cusum["peak"]

        # --- controller response -----------------------------------------
        demand = case.demand[car]
        cooling_rows = int(cooling.sum())
        feats["full_demand_frac"] = (float((demand == 2).sum()) / cooling_rows
                                     if cooling_rows > 0 else float("nan"))

        # --- integrity ----------------------------------------------------
        reporting = max(1, int(case.reporting[car].sum()))
        bad = int(case.invalid[car].sum()) + int(case.dropout[car].sum())
        # log1p on a per-mille rate: the channel is extremely sparse (tens of
        # rows out of thousands) so its raw scale would be meaningless, but the
        # ordering between cars is what carries the evidence.
        feats["integrity_loss"] = float(np.log1p(1000.0 * bad / reporting))
        feats["_integrity_events"] = float(bad)

        out[car] = feats
    return out


# --------------------------------------------------------------------------
# refrigerant-circuit branch
# --------------------------------------------------------------------------
def refrigerant_features(case: CleanCase) -> dict[str, dict[str, float]]:
    """
    Undercharge indicators from the two refrigeration circuits.

    The key move is to use the *sibling circuit inside the same pack* as the
    reference. Both circuits see the same ambient, the same cabin and the same
    demand command, so any systematic difference between them is a property of
    the refrigerant loop itself - which removes the duty-cycle confound that
    makes raw pressure comparisons between cars misleading.
    """
    out: dict[str, dict[str, float]] = {}
    if not case.circuits:
        return out

    dt = case.panel.dt_seconds
    per_car_circuits: dict[str, dict[int, dict]] = {}
    for car in case.cars:
        per_car_circuits[car] = {}
        for k, channel in case.circuits.items():
            if car not in channel["p_low"].columns:
                continue
            # Restrict to rows where *both* circuits of this car are running, so
            # the intra-pack comparison is strictly like-for-like.
            both_running = None
            for kk, ch in case.circuits.items():
                run = ch["running"][car] if car in ch["running"].columns else None
                if run is None:
                    continue
                both_running = run if both_running is None else (both_running & run)
            rows = both_running if both_running is not None else channel["running"][car]
            ind = phys.circuit_indicators(channel["p_low"][car].where(rows),
                                          channel["p_high"][car].where(rows))
            ind["duty"] = _safe(channel["comp"][car].where(case.cooling[car]).mean())
            ind["cycling"] = _rate_per_hour(channel["comp"][car], dt)
            ind["lift_stratified"] = _strat_mean(
                (channel["p_high"][car] - channel["p_low"][car]).where(rows),
                case.demand[car].where(rows))
            per_car_circuits[car][k] = ind

    # sibling-car reference for the weakest circuit's pressure lift
    weakest_lift = {car: np.nanmin([c["lift_stratified"] for c in ch.values()])
                    if ch else float("nan")
                    for car, ch in per_car_circuits.items()}
    peer_lift = phys.nan_median(list(weakest_lift.values()))

    for car, channels in per_car_circuits.items():
        feats: dict[str, float] = {}
        if len(channels) >= 2:
            keys = sorted(channels)
            a, b = channels[keys[0]], channels[keys[1]]
            scale_lift = np.nanmean([a["lift"], b["lift"]])
            scale_high = np.nanmean([a["p_high_med"], b["p_high_med"]])
            feats["circuit_asym_lift"] = _safe(abs(a["lift"] - b["lift"]) / scale_lift) \
                if np.isfinite(scale_lift) and scale_lift > EPS else float("nan")
            feats["circuit_asym_high"] = _safe(abs(a["p_high_med"] - b["p_high_med"]) / scale_high) \
                if np.isfinite(scale_high) and scale_high > EPS else float("nan")
            feats["circuit_asym_ratio"] = _safe(abs(a["ratio"] - b["ratio"]))
        else:
            feats["circuit_asym_lift"] = float("nan")
            feats["circuit_asym_high"] = float("nan")
            feats["circuit_asym_ratio"] = float("nan")

        excursions = [c["suction_excursion"] for c in channels.values()]
        sd = [c["suction_sd"] for c in channels.values()]
        scale = phys.nan_median([c["p_low_med"] for c in channels.values()])
        if np.isfinite(scale) and scale > EPS:
            feats["suction_excursion"] = _safe(np.nanmax(excursions) / scale)
        else:
            feats["suction_excursion"] = _safe(np.nanmax(excursions)) if excursions else float("nan")
        feats["_suction_sd"] = _safe(np.nanmax(sd)) if sd else float("nan")

        lift = weakest_lift.get(car, float("nan"))
        feats["lift_deficit"] = _safe((peer_lift - lift) / peer_lift) \
            if np.isfinite(peer_lift) and peer_lift > EPS else float("nan")
        feats["_weakest_lift"] = _safe(lift)

        feats["compressor_duty"] = _safe(np.nanmean([c["duty"] for c in channels.values()])) \
            if channels else float("nan")
        feats["compressor_cycling"] = _safe(np.nanmax([c["cycling"] for c in channels.values()])) \
            if channels else float("nan")
        out[car] = feats
    return out


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------
def extract(case: CleanCase) -> tuple[pd.DataFrame, dict]:
    """
    Return ``(features, context)`` where ``features`` is a cars x features frame
    restricted to the columns declared in :data:`acv2.config.FEATURE_SPEC`, plus
    the underscore-prefixed diagnostics used for explanations.
    """
    ctx = build_context(case)
    thermal = thermal_features(case, ctx)
    refrigerant = refrigerant_features(case)

    merged: dict[str, dict[str, float]] = {}
    for car in ctx["cars"]:
        row = dict(thermal.get(car, {}))
        row.update(refrigerant.get(car, {}))
        merged[car] = row

    frame = pd.DataFrame(merged).T if merged else pd.DataFrame()
    # Guarantee every declared feature exists, so downstream code never has to
    # test for a schema-dependent column.
    for name in cfg.FEATURE_SPEC:
        if name not in frame.columns:
            frame[name] = np.nan
    ordered = list(cfg.FEATURE_SPEC) + sorted(c for c in frame.columns if c.startswith("_"))
    frame = frame.reindex(columns=ordered)
    frame.index.name = "car"

    ctx["has_refrigerant"] = bool(refrigerant)
    ctx["active_groups"] = sorted({
        cfg.FEATURE_SPEC[name][0] for name in cfg.FEATURE_SPEC
        if name in frame.columns and frame[name].notna().sum() >= 2
        and frame[name].nunique(dropna=True) > 1
    })
    return frame, ctx
