"""
acv_physics_ranker.py
==============================================================================
Physics-informed refrigerant-leakage localisation for train ACV consists.

METHOD (single engine, fully deterministic, nothing fitted)
--------------------------------------------------------------------------
A refrigerant leak reduces the mass of working fluid circulating through the
evaporator. Cooling capacity falls, so the affected car's cabin drifts above
its own setpoint and above its sibling cars -- even though every car on the
consist shares identical outdoor weather, solar load, train speed and door
cycles. That shared environment is what makes the *cross-car* comparison
valid: the median cabin temperature of the co-cooling cars is a live
environmental reference, so subtracting it cancels the dominant nuisance
variable without needing a single fitted parameter.

    dT_rel(c,t) = T_in(c,t) - median over peers k that are ALSO cooling at t
    dT_set(c,t) = T_in(c,t) - T_cooling_setpoint(c,t)

Five same-signed statistics of those two signals are combined into one health
index by fixed a-priori weights. Higher index = more likely leaking. Cars are
then ranked in descending index order.

Why not an unsupervised anomaly detector: an anomaly score is direction-blind
(it flags "different", not "too warm to hold setpoint"), it has no peer
reference to cancel weather, and with only 6 labelled journeys nothing learned
can be validated. The physics index needs no training data at all.

Only the 8 SCADA parameters that exist in EVERY case file are used, so there
is exact parity between the development cases and the held-out test case. The
richer per-car refrigeration telemetry present in acv_case_04 (compressor
high/low pressures, per-circuit run states) is deliberately NOT used, because
it is absent from the test file.
==============================================================================
"""

from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Schema harmonisation. Rolling-stock generations name the same physical
# quantity differently; map each alias onto one canonical role.
# ---------------------------------------------------------------------------
ROLE_ALIASES: dict[str, tuple[str, ...]] = {
    "indoor": (
        "Indoor Average Temperature",
        "Passenger Cabin Temperature Detected Value",
    ),
    "outdoor": (
        "Outdoor Average Temperature",
        "Outside Temperature Sensor Reading",
        "Fresh Air Temperature Detected Value",
    ),
    "set_cool": (
        "ACV Control Temperature (Cooling)",
        "Target Temperature Value",
    ),
    "run_mode": (
        "ACV Running Mode",
    ),
    # Only a genuine data-validity flag belongs here. 'ACV Operating Mode'
    # looks tempting because it also emits the token 'Invalid', but it is a
    # mode word, not a sensor-health flag: in acv_case_04 it reads 'Invalid'
    # on the overwhelming majority of rows, and treating it as a validity gate
    # discards the entire journey. Files without this flag are simply not gated.
    "valid": (
        "ACV Information Valid",
    ),
}

CAR_COL_RE = re.compile(r"^Car\s+(\S+)\s+-\s+(.*)$")

# Physically impossible cabin readings: CAN-bus dropouts arrive as 0.0 C.
TEMP_MIN_C = 5.0
TEMP_MAX_C = 45.0

# A car is "actively cooling" when its running mode names a cooling regime.
# Covers 'Automatic Cooling', 'Full Cooling', 'Half Cooling', plain 'Cooling'.
COOLING_TOKEN = "cooling"

# Elevation that counts as a meaningful thermal deficit.
REL_ELEVATION_C = 0.20

# Minimum number of simultaneously-cooling cars for a timestamp to yield a
# usable peer median. Below this the "peer group" is too thin to be a baseline.
MIN_PEERS_COOLING = 3

# Persistence window: a leak is a sustained capacity loss, not a transient.
PERSISTENCE_MINUTES = 15.0

# Fixed a-priori weights. dT_rel is the primary evidence; the rest corroborate
# it from different angles (own-setpoint error, peak-load tail, duty fraction,
# temporal continuity). These are engineering judgement, NOT fitted values.
PHYSICS_WEIGHTS: dict[str, float] = {
    "mean_rel_cooling":  1.00,   # mean elevation above co-cooling peers
    "mean_t_minus_set":  0.50,   # mean failure to reach its own setpoint
    "p95_rel":           0.25,   # peak-load excursion tail
    "frac_rel_elevated": 0.25,   # fraction of cooling time elevated
    "persistence_15m":   0.25,   # sustained, not transient
}

FEATURE_ORDER = list(PHYSICS_WEIGHTS.keys())


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _car_columns(columns) -> dict[tuple[str, str], str]:
    """Map (car_id, role) -> actual column name, for recognised roles only."""
    alias_to_role = {
        alias: role for role, aliases in ROLE_ALIASES.items() for alias in aliases
    }
    found: dict[tuple[str, str], str] = {}
    for col in columns:
        text = str(col)
        if text == "Car model":
            continue
        m = CAR_COL_RE.match(text)
        if not m:
            continue
        car_id, param = m.group(1).strip(), m.group(2).strip()
        role = alias_to_role.get(param)
        if role is None:
            continue
        # First alias listed for a role wins, so 'ACV Information Valid'
        # takes precedence over the 'ACV Operating Mode' fallback.
        if (car_id, role) not in found:
            found[(car_id, role)] = col
    return found


def header_car_ids(columns) -> list[str]:
    """Every car identifier appearing in the file's own headers, as written."""
    ids = set()
    for col in columns:
        text = str(col)
        if text == "Car model":
            continue
        m = CAR_COL_RE.match(text)
        if m:
            ids.add(m.group(1).strip())
    return sorted(ids)


def _time_column(columns) -> str | None:
    for col in columns:
        if not str(col).startswith("Car ") and "ime" in str(col):
            return col
    return None


def _excel_engine() -> str | None:
    """
    Prefer the calamine engine when installed. acv_case_04.xlsx is 33 MB with
    483 columns and openpyxl needs ~5 minutes on it; calamine reads it in
    seconds. Returns None so pandas picks its default when calamine is absent,
    which changes nothing except run time.
    """
    try:
        import python_calamine  # noqa: F401
        return "calamine"
    except ImportError:
        return None


def load_case(path: str) -> tuple[pd.DataFrame, dict[tuple[str, str], str], list[str], str | None]:
    """
    Read only the columns the physics model needs.

    acv_case_04.xlsx is 33 MB / 483 columns; reading it whole is slow and
    pointless, so the header is inspected first and the body is read with
    usecols restricted to the ~40 relevant columns.
    """
    is_csv = path.lower().endswith(".csv")
    engine = None if is_csv else _excel_engine()

    if is_csv:
        head = pd.read_csv(path, nrows=0)
    else:
        head = pd.read_excel(path, nrows=0, engine=engine)

    colmap = _car_columns(head.columns)
    all_ids = header_car_ids(head.columns)
    tcol = _time_column(head.columns)

    needed = sorted({c for c in colmap.values()})
    if tcol is not None:
        needed = [tcol] + needed

    if is_csv:
        df = pd.read_csv(path, usecols=needed)
    else:
        df = pd.read_excel(path, usecols=needed, engine=engine)

    return df, colmap, all_ids, tcol


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def clean_indoor_temperatures(
    df: pd.DataFrame,
    colmap: dict[tuple[str, str], str],
    car_ids: list[str],
) -> pd.DataFrame:
    """
    Return a (timestamp x car) frame of trustworthy cabin temperatures.

    Two defects are removed, both by masking to NaN rather than by filling:
      1. rows the ACV itself flags 'Invalid' -- the sensor is not reporting;
      2. readings outside 5..45 C -- CAN-bus dropouts that arrive as 0.0 C.

    Nothing is forward-filled. Interpolating across an overnight depot gap
    would fabricate cooling-window data, and every downstream statistic is
    NaN-aware anyway, so filling buys nothing and can only mislead.
    """
    out = {}
    for cid in car_ids:
        icol = colmap.get((cid, "indoor"))
        if icol is None:
            continue
        s = pd.to_numeric(df[icol], errors="coerce")

        vcol = colmap.get((cid, "valid"))
        if vcol is not None:
            invalid = df[vcol].astype(str).str.contains("invalid", case=False, na=False)
            s = s.mask(invalid)

        s = s.mask((s <= TEMP_MIN_C) | (s >= TEMP_MAX_C))
        out[cid] = s

    indoor = pd.DataFrame(out, index=df.index)
    # Depot / power-off rows: nobody reporting anything.
    return indoor.dropna(how="all")


def cooling_mask(
    df: pd.DataFrame,
    colmap: dict[tuple[str, str], str],
    car_ids: list[str],
    index: pd.Index,
) -> pd.DataFrame:
    """(timestamp x car) boolean: is this car in an active cooling regime?"""
    out = {}
    for cid in car_ids:
        mcol = colmap.get((cid, "run_mode"))
        if mcol is None:
            # No mode telemetry: assume the unit is cooling, and let the
            # temperature statistics speak for themselves.
            out[cid] = pd.Series(True, index=df.index)
        else:
            out[cid] = df[mcol].astype(str).str.contains(
                COOLING_TOKEN, case=False, na=False
            )
    return pd.DataFrame(out, index=df.index).loc[index]


def setpoints(
    df: pd.DataFrame,
    colmap: dict[tuple[str, str], str],
    car_ids: list[str],
    index: pd.Index,
) -> pd.DataFrame:
    out = {}
    for cid in car_ids:
        scol = colmap.get((cid, "set_cool"))
        out[cid] = (
            pd.to_numeric(df[scol], errors="coerce")
            if scol is not None
            else pd.Series(np.nan, index=df.index)
        )
    sp = pd.DataFrame(out, index=df.index)
    # Setpoints are physical temperatures too; reject impossible values.
    sp = sp.mask((sp <= TEMP_MIN_C) | (sp >= TEMP_MAX_C))
    return sp.loc[index]


# ---------------------------------------------------------------------------
# Persistence helper
# ---------------------------------------------------------------------------
def _segment_ids(times: pd.Series | None, n: int) -> tuple[np.ndarray, float]:
    """
    Split the record into contiguous acquisition segments and report the
    median sampling interval in seconds.

    Journeys span several days with overnight gaps. A rolling mean must not
    straddle those gaps, or a 15-minute window silently becomes 15 hours.
    """
    if times is None:
        return np.zeros(n, dtype=int), 30.0
    t = pd.to_datetime(times, errors="coerce")
    dt = t.diff().dt.total_seconds()
    median_dt = float(dt.median()) if dt.notna().any() else 30.0
    if not np.isfinite(median_dt) or median_dt <= 0:
        median_dt = 30.0
    breaks = (dt > 3.0 * median_dt).fillna(False).to_numpy()
    return np.cumsum(breaks).astype(int), median_dt


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_features(path: str) -> dict:
    df, colmap, all_ids, tcol = load_case(path)

    indoor = clean_indoor_temperatures(df, colmap, all_ids)
    if indoor.empty:
        raise ValueError(f"No usable cabin temperature telemetry in {path}")

    idx = indoor.index
    cooling = cooling_mask(df, colmap, all_ids, idx)
    sp = setpoints(df, colmap, all_ids, idx)

    # A car is assessable only if it actually reports cabin temperature.
    active = [c for c in indoor.columns if indoor[c].notna().any()]
    indoor, cooling, sp = indoor[active], cooling[active], sp[active]

    # --- cross-car baseline, restricted to cars that are cooling right now ---
    # Comparing a cooling car against a ventilating one is not a like-for-like
    # comparison, so peers are masked to the co-cooling set at each timestamp.
    indoor_when_cooling = indoor.where(cooling)
    n_cooling = indoor_when_cooling.notna().sum(axis=1)
    enough_peers = n_cooling >= min(MIN_PEERS_COOLING, max(1, len(active)))

    peer_median = indoor_when_cooling.median(axis=1, skipna=True)
    rel = indoor_when_cooling.sub(peer_median, axis=0).where(enough_peers, np.nan)
    t_minus_set = indoor_when_cooling.sub(sp).where(enough_peers, np.nan)

    seg, median_dt = _segment_ids(df[tcol] if tcol else None, len(df))
    seg = pd.Series(seg, index=df.index).loc[idx]
    window = max(2, int(round(PERSISTENCE_MINUTES * 60.0 / median_dt)))

    rows = {}
    for c in active:
        r = rel[c]
        r_valid = r.dropna()
        tms = t_minus_set[c].dropna()

        # 15-minute rolling mean, restarted at every acquisition gap.
        roll = r.groupby(seg).transform(
            lambda s: s.rolling(window, min_periods=max(2, window // 3)).mean()
        )
        roll_valid = roll.dropna()

        rows[c] = {
            "mean_rel_cooling":  float(r_valid.mean()) if len(r_valid) else 0.0,
            "mean_t_minus_set":  float(tms.mean()) if len(tms) else 0.0,
            "p95_rel":           float(np.percentile(r_valid, 95)) if len(r_valid) else 0.0,
            "frac_rel_elevated": float((r_valid > REL_ELEVATION_C).mean()) if len(r_valid) else 0.0,
            "persistence_15m":   float((roll_valid > REL_ELEVATION_C).mean()) if len(roll_valid) else 0.0,
            "n_cooling_samples": int(len(r_valid)),
            "mean_indoor_c":     float(indoor[c].mean()),
        }

    feats = pd.DataFrame(rows).T.reindex(active)
    return {
        "features": feats,
        "active_cars": active,
        "all_cars": all_ids,
        "median_dt_s": median_dt,
        "persistence_window_rows": window,
        "n_rows": int(len(idx)),
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def robust_z(values: pd.Series) -> pd.Series:
    """
    Median / MAD standardisation within the consist.

    Mean-and-sigma standardisation is contaminated by the very outlier being
    looked for: one leaking car inflates sigma and drags the mean towards
    itself, shrinking its own z-score. Median and MAD are unaffected by a
    single outlier among eight cars, so the faulty car separates further.
    """
    v = values.astype(float)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826
    if mad > 1e-9:
        return (v - med) / mad
    sd = float(v.std(ddof=0))
    if sd > 1e-9:
        return (v - v.mean()) / sd
    return pd.Series(0.0, index=v.index)


def rank_consist(path: str, weights: dict[str, float] | None = None) -> dict:
    """Localise the leaking car and rank every car in the file."""
    w = dict(PHYSICS_WEIGHTS if weights is None else weights)
    ex = extract_features(path)
    feats: pd.DataFrame = ex["features"]
    active: list[str] = ex["active_cars"]

    z = pd.DataFrame({f: robust_z(feats[f]) for f in FEATURE_ORDER}, index=feats.index)
    index_score = sum(w[f] * z[f] for f in FEATURE_ORDER)
    index_score = index_score.sort_values(ascending=False)

    ranked = index_score.index.tolist()
    # Cars present in the headers but reporting no temperature cannot be
    # assessed; they go to the tail so the output is still a full permutation.
    unassessable = [c for c in ex["all_cars"] if c not in active]
    ranked_full = ranked + unassessable

    margin = (
        float(index_score.iloc[0] - index_score.iloc[1])
        if len(index_score) > 1
        else float("nan")
    )

    diagnostics = []
    for r, c in enumerate(ranked_full, start=1):
        if c in active:
            diagnostics.append({
                "rank": r,
                "car": c,
                "health_index": round(float(index_score[c]), 3),
                "mean_rel_c": round(float(feats.loc[c, "mean_rel_cooling"]), 3),
                "mean_t_minus_set_c": round(float(feats.loc[c, "mean_t_minus_set"]), 3),
                "p95_rel_c": round(float(feats.loc[c, "p95_rel"]), 3),
                "frac_elevated_pct": round(float(feats.loc[c, "frac_rel_elevated"]) * 100, 1),
                "persistence_pct": round(float(feats.loc[c, "persistence_15m"]) * 100, 1),
                "n_cooling_samples": int(feats.loc[c, "n_cooling_samples"]),
                "assessable": True,
            })
        else:
            diagnostics.append({
                "rank": r, "car": c, "health_index": None, "assessable": False,
            })

    top = ranked_full[0]
    summary = (
        f"Car {top} is the most likely refrigerant-leak location. During active "
        f"cooling it ran {feats.loc[top, 'mean_rel_cooling']:+.2f} degC relative to the "
        f"co-cooling consist median, sat {feats.loc[top, 'mean_t_minus_set']:+.2f} degC "
        f"from its own cooling setpoint, and stayed more than "
        f"{REL_ELEVATION_C:.2f} degC above its peers for "
        f"{feats.loc[top, 'persistence_15m'] * 100:.0f}% of the time on a "
        f"{PERSISTENCE_MINUTES:.0f}-minute rolling basis. Separation from the "
        f"second-ranked car is {margin:.2f} index points."
    ) if top in active else f"Car {top} ranked first but reports no usable telemetry."

    return {
        "file_id": os.path.basename(path),
        "ranked_cars": "|".join(ranked_full),
        "ranked_cars_list": ranked_full,
        "most_likely_faulty_car": top,
        "confidence_margin": round(margin, 3) if np.isfinite(margin) else None,
        "consist_size": len(ranked_full),
        "active_cars": active,
        "unassessable_cars": unassessable,
        "features": feats,
        "health_index": index_score,
        "car_diagnostics": diagnostics,
        "diagnostic_summary": summary,
        "median_dt_s": ex["median_dt_s"],
        "persistence_window_rows": ex["persistence_window_rows"],
        "n_rows": ex["n_rows"],
    }


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "PS3", "02_Datasets", "ACV",
        "Test", "acv_test_case.xlsx",
    )
    res = rank_consist(target)
    print(f"file            : {res['file_id']}")
    print(f"ranked_cars     : {res['ranked_cars']}")
    print(f"most likely leak: Car {res['most_likely_faulty_car']}")
    print(f"margin          : {res['confidence_margin']}")
    print(f"sampling        : {res['median_dt_s']:.0f}s, "
          f"persistence window {res['persistence_window_rows']} rows")
    print()
    print(pd.DataFrame(res["car_diagnostics"]).to_string(index=False))
    print()
    print(res["diagnostic_summary"])
