"""
backend/models/acv_model.py
===============================================================================
Air Conditioning & Ventilation (ACV) Subsystem Model.

Two distinct capabilities live here:

1. evaluate_telemetry(...)
   The live 10 Hz digital-twin path. One rooftop ACV pack, evaluated for
   thermodynamic heat exchange, COP efficiency and filter obstruction.
   Called by InferenceBroker on every frame. Unchanged behaviour.

2. localise_leak(...) / rank_consist_window(...)
   The real refrigerant-leakage diagnosis model, ported from the ACV
   subsystem work. Given per-car cabin temperatures across a consist, it
   ranks every car from most to least likely to be leaking refrigerant.

There is deliberately NO model artefact to load. The previous version of this
file loaded acv_model_bundle.joblib but never used it; the pickle held an L2
logistic pairwise ranker whose measured contribution to ranking accuracy was
exactly zero (0.9583 with it, 0.9583 without). It has been replaced by the
closed-form physics index below, which has no fitted parameters, cannot suffer
scikit-learn version drift, and needs no training data.

-------------------------------------------------------------------------------
PHYSICS
-------------------------------------------------------------------------------
A refrigerant leak reduces the mass of working fluid circulating through the
evaporator, so cooling capacity falls and the affected cabin drifts above its
setpoint and above its sibling cars. Every car on a consist shares the same
outdoor weather, solar load, speed and door cycles at every instant, which is
what licenses the cross-car comparison: the median cabin temperature of the
cars cooling at that same timestamp is a live environmental reference, and
subtracting it removes the dominant nuisance variable with no fitted terms.

    dT_rel(c,t) = T_in(c,t) - median over cars k also cooling at t
    dT_set(c,t) = T_in(c,t) - T_cooling_setpoint(c,t)

dT_rel is peer-relative and immune to a consist-wide hot day. dT_set is
absolute and survives a consist-wide degradation. Using both means neither
failure mode is blind.

-------------------------------------------------------------------------------
NOTE ON CODE DUPLICATION
-------------------------------------------------------------------------------
The scoring core below is an intentional standalone copy of
ACV/acv_physics_ranker.py, which remains the reference implementation and the
one wired into the hackathon deliverable ACV/predict.py. The copy keeps the
backend importable without depending on a path outside the backend package.
Both copies are verified to agree numerically (see
backend/models/test_acv_leak_model.py). If the physics constants or weights
below are edited, edit ACV/acv_physics_ranker.py to match, and re-run that
test.
===============================================================================
"""

import os
import re

import numpy as np

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:                                      # pragma: no cover
    _PANDAS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Schema harmonisation. Rolling-stock generations name the same physical
# quantity differently; map each alias onto one canonical role.
# ---------------------------------------------------------------------------
ROLE_ALIASES = {
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
    # also emits the token 'Invalid' but is a mode word, not a sensor-health
    # flag: treating it as a validity gate discards entire journeys.
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

# Elevation above the consist that counts as a meaningful thermal deficit.
REL_ELEVATION_C = 0.20

# Minimum simultaneously-cooling cars for a timestamp to give a usable peer
# median. Below this the peer group is too thin to be a baseline.
MIN_PEERS_COOLING = 3

# A leak is a sustained capacity loss, not a transient.
PERSISTENCE_MINUTES = 15.0

# Fixed a-priori weights. dT_rel is the primary evidence; the rest corroborate
# from different angles. Engineering judgement, NOT fitted values.
PHYSICS_WEIGHTS = {
    "mean_rel_cooling":  1.00,   # mean elevation above co-cooling peers
    "mean_t_minus_set":  0.50,   # mean failure to reach its own setpoint
    "p95_rel":           0.25,   # peak-load excursion tail
    "frac_rel_elevated": 0.25,   # fraction of cooling time elevated
    "persistence_15m":   0.25,   # sustained, not transient
}

FEATURE_ORDER = list(PHYSICS_WEIGHTS.keys())

# Plain-language copy for the HUD / AI insight layer.
FEATURE_LABELS = {
    "mean_rel_cooling":  "Warmer than the other cars",
    "mean_t_minus_set":  "Missing its own target temperature",
    "p95_rel":           "Worst-case warm excursion",
    "frac_rel_elevated": "How often it runs warm",
    "persistence_15m":   "How long it stays warm",
}


# ---------------------------------------------------------------------------
# Column discovery
# ---------------------------------------------------------------------------
def _car_columns(columns):
    """Map (car_id, role) -> actual column name, for recognised roles only."""
    alias_to_role = {
        alias: role for role, aliases in ROLE_ALIASES.items() for alias in aliases
    }
    found = {}
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
        if (car_id, role) not in found:
            found[(car_id, role)] = col
    return found


def header_car_ids(columns):
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


def _time_column(columns):
    for col in columns:
        if not str(col).startswith("Car ") and "ime" in str(col):
            return col
    return None


def _excel_engine():
    """Prefer calamine when installed; it reads the large cases far faster."""
    try:
        import python_calamine  # noqa: F401
        return "calamine"
    except ImportError:
        return None


def load_case(path: str):
    """Read only the columns the physics model needs."""
    is_csv = str(path).lower().endswith(".csv")
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
def clean_indoor_temperatures(df, colmap, car_ids):
    """
    Return a (timestamp x car) frame of trustworthy cabin temperatures.

    Defects are removed by masking to NaN, never by filling. Journeys span
    several days with overnight gaps; interpolating across them would fabricate
    cooling-window data. Every downstream statistic is NaN-aware.
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


def cooling_mask(df, colmap, car_ids, index):
    """(timestamp x car) boolean: is this car in an active cooling regime?"""
    out = {}
    for cid in car_ids:
        mcol = colmap.get((cid, "run_mode"))
        if mcol is None:
            # No mode telemetry: assume cooling and let the temperatures speak.
            out[cid] = pd.Series(True, index=df.index)
        else:
            out[cid] = df[mcol].astype(str).str.contains(
                COOLING_TOKEN, case=False, na=False
            )
    return pd.DataFrame(out, index=df.index).loc[index]


def setpoint_frame(df, colmap, car_ids, index):
    out = {}
    for cid in car_ids:
        scol = colmap.get((cid, "set_cool"))
        out[cid] = (
            pd.to_numeric(df[scol], errors="coerce")
            if scol is not None
            else pd.Series(np.nan, index=df.index)
        )
    sp = pd.DataFrame(out, index=df.index)
    sp = sp.mask((sp <= TEMP_MIN_C) | (sp >= TEMP_MAX_C))
    return sp.loc[index]


def _segment_ids(times, n):
    """
    Split the record into contiguous acquisition segments and report the median
    sampling interval. A rolling mean must not straddle an overnight gap, or a
    15-minute window silently becomes 15 hours.
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
# Scoring core
# ---------------------------------------------------------------------------
def features_from_frames(indoor, cooling, setpts=None, median_dt_s=30.0, segments=None):
    """
    Compute the five physics features from aligned (timestamp x car) frames.

    Kept independent of file I/O so the same core serves both an offline case
    file and a live rolling telemetry buffer.
    """
    active = [c for c in indoor.columns if indoor[c].notna().any()]
    if not active:
        return pd.DataFrame(columns=FEATURE_ORDER + ["n_cooling_samples", "mean_indoor_c"])

    indoor = indoor[active]
    cooling = cooling.reindex(columns=active, fill_value=True).reindex(indoor.index)
    if setpts is None:
        setpts = pd.DataFrame(np.nan, index=indoor.index, columns=active)
    else:
        setpts = setpts.reindex(columns=active).reindex(indoor.index)

    if segments is None:
        segments = pd.Series(0, index=indoor.index)
    else:
        segments = segments.reindex(indoor.index).fillna(0)

    # Cross-car baseline restricted to cars cooling at this same timestamp.
    indoor_when_cooling = indoor.where(cooling)
    n_cooling = indoor_when_cooling.notna().sum(axis=1)
    enough_peers = n_cooling >= min(MIN_PEERS_COOLING, max(1, len(active)))

    peer_median = indoor_when_cooling.median(axis=1, skipna=True)
    rel = indoor_when_cooling.sub(peer_median, axis=0).where(enough_peers, np.nan)
    t_minus_set = indoor_when_cooling.sub(setpts).where(enough_peers, np.nan)

    window = max(2, int(round(PERSISTENCE_MINUTES * 60.0 / max(median_dt_s, 1e-6))))

    rows = {}
    for c in active:
        r = rel[c]
        r_valid = r.dropna()
        tms = t_minus_set[c].dropna()

        roll = r.groupby(segments).transform(
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

    return pd.DataFrame(rows).T.reindex(active)


def robust_z(values):
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


def score_features(feats, weights=None):
    """Weighted robust-z health index per car, sorted most to least suspect."""
    w = dict(PHYSICS_WEIGHTS if weights is None else weights)
    if feats.empty:
        return pd.Series(dtype=float)
    z = pd.DataFrame({f: robust_z(feats[f]) for f in FEATURE_ORDER}, index=feats.index)
    return sum(w[f] * z[f] for f in FEATURE_ORDER).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Subsystem model
# ---------------------------------------------------------------------------
class ACVSubsystemModel:
    """
    ACV subsystem model.

    Live twin path  : evaluate_telemetry()
    Diagnosis path  : localise_leak() / rank_consist_window()
    """

    #: Reported to the API so the UI can state which method is in use.
    METHOD = "physics_informed_cross_car_thermal_deficit"
    METHOD_VERSION = "2.0.0"

    def __init__(self):
        # No artefact to load: the ranker is closed-form. Kept as an explicit
        # attribute so callers can branch on capability rather than guessing.
        self.leak_model_available = _PANDAS_AVAILABLE
        self.physics_weights = dict(PHYSICS_WEIGHTS)
        self.thresholds = {
            "temp_min_c": TEMP_MIN_C,
            "temp_max_c": TEMP_MAX_C,
            "rel_elevation_c": REL_ELEVATION_C,
            "min_peers_cooling": MIN_PEERS_COOLING,
            "persistence_minutes": PERSISTENCE_MINUTES,
        }

    # -- capability description -------------------------------------------
    def describe_method(self) -> dict:
        """Metadata for /api/health and the AI insight layer."""
        return {
            "method": self.METHOD,
            "version": self.METHOD_VERSION,
            "fitted_parameters": 0,
            "requires_training_data": False,
            "model_artefact": None,
            "leak_localisation_available": self.leak_model_available,
            "physics_weights": dict(self.physics_weights),
            "thresholds": dict(self.thresholds),
            "feature_labels": dict(FEATURE_LABELS),
            "summary": (
                "Ranks every car in a consist by a weighted robust-z index of five "
                "thermal-deficit statistics, measured against the median of the cars "
                "cooling at the same instant. No fitted parameters."
            ),
        }

    # -- live 10 Hz twin path ---------------------------------------------
    def evaluate_telemetry(self, supply_temp: float, return_temp: float,
                           compressor_kw: float, degraded: bool = False) -> dict:
        delta_temp = round(float(return_temp - supply_temp), 1)

        # Baseline COP efficiency rating.
        # Nominal delta is ~7.5 - 9.0 deg C at 4.0 - 5.0 kW.
        nominal_delta = 8.5
        efficiency_ratio = delta_temp / max(nominal_delta, 1.0)
        power_ratio = compressor_kw / 4.5

        if degraded:
            efficiency_rating = round(float(np.clip(efficiency_ratio * 0.72, 0.40, 0.75)), 2)
            anomaly_score = round(float(np.clip(
                (1.0 - efficiency_rating) * 1.3 + (power_ratio - 1.0) * 0.3, 0.55, 0.95)), 2)
            fault_type = "REFRIGERANT_LEAKAGE" if delta_temp < 6.0 else "FILTER_CLOGGING"
        else:
            efficiency_rating = round(float(np.clip(efficiency_ratio * 0.95, 0.75, 0.99)), 2)
            anomaly_score = round(float(np.clip(1.0 - efficiency_rating, 0.05, 0.35)), 2)
            fault_type = "NONE"

        return {
            "unit_id": "ACV_PACK_1",
            "supply_temp_c": round(float(supply_temp), 1),
            "return_temp_c": round(float(return_temp), 1),
            "delta_temp_c": delta_temp,
            "compressor_power_kw": round(float(compressor_kw), 2),
            "efficiency_rating": efficiency_rating,
            "anomaly_score": anomaly_score,
            "fault_type": fault_type,
        }

    # -- consist leak localisation ----------------------------------------
    def localise_leak(self, source, file_id: str = None) -> dict:
        """
        Rank every car in an ACV consist from most to least likely to be
        leaking refrigerant.

        source:  a path to an .xlsx/.csv consist file in the PS3 ACV format,
                 OR an already-loaded pandas DataFrame in that same format
                 (used by the /api upload route, which parses CSV bytes first).
        file_id: label echoed back in the result; defaults to the file's base
                 name, or 'uploaded_data' for a DataFrame.

        Returns ranked_cars (pipe separated, as the submission format wants),
        the leading suspect, the separation from the runner-up, and a per-car
        diagnostic table suitable for direct display in the HUD.
        """
        if not self.leak_model_available:
            raise RuntimeError("pandas is required for ACV leak localisation")

        if isinstance(source, pd.DataFrame):
            df = source
            colmap = _car_columns(df.columns)
            all_ids = header_car_ids(df.columns)
            tcol = _time_column(df.columns)
            label = file_id or "uploaded_data"
        else:
            if not os.path.exists(source):
                raise FileNotFoundError(source)
            df, colmap, all_ids, tcol = load_case(source)
            label = file_id or os.path.basename(source)

        if not colmap:
            raise ValueError(
                f"{label}: no recognised ACV columns. Expected headers of the form "
                f"'Car <NN> - Indoor Average Temperature'."
            )

        indoor = clean_indoor_temperatures(df, colmap, all_ids)
        if indoor.empty:
            raise ValueError(f"No usable cabin temperature telemetry in {label}")

        idx = indoor.index
        cooling = cooling_mask(df, colmap, all_ids, idx)
        sp = setpoint_frame(df, colmap, all_ids, idx)
        seg_arr, median_dt = _segment_ids(df[tcol] if tcol else None, len(df))
        seg = pd.Series(seg_arr, index=df.index).loc[idx]

        feats = features_from_frames(indoor, cooling, sp, median_dt, seg)
        index_score = score_features(feats)

        return self._assemble(
            feats, index_score, all_ids,
            file_id=label,
            median_dt_s=median_dt,
            n_rows=int(len(idx)),
        )

    def rank_consist_window(self, cabin_temps: dict, setpoints: dict = None,
                            cooling_flags: dict = None, sample_interval_s: float = 30.0) -> dict:
        """
        Same index, driven by a live rolling buffer instead of a file.

        cabin_temps:   {car_id: [T0, T1, ...]} equal-length cabin temperature
                       series, oldest first
        setpoints:     {car_id: [...]} or {car_id: scalar}; omitted -> the
                       setpoint term contributes nothing
        cooling_flags: {car_id: [bool, ...]} or {car_id: bool}; omitted ->
                       every sample is treated as active cooling
        sample_interval_s: seconds between samples, used to size the
                       15-minute persistence window correctly

        Intended for a consist-aware live twin. The current single-pack twin
        does not emit per-car cabin temperatures, so nothing in the 10 Hz path
        calls this yet; it exists so the streaming layer can adopt consist
        telemetry without reimplementing the physics.
        """
        if not self.leak_model_available:
            raise RuntimeError("pandas is required for ACV leak localisation")
        if not cabin_temps:
            raise ValueError("cabin_temps is empty")

        indoor = pd.DataFrame({c: pd.Series(v, dtype=float) for c, v in cabin_temps.items()})
        indoor = indoor.mask((indoor <= TEMP_MIN_C) | (indoor >= TEMP_MAX_C))

        def expand(src, default):
            if src is None:
                return pd.DataFrame(default, index=indoor.index, columns=indoor.columns)
            cols = {}
            for c in indoor.columns:
                v = src.get(c, default)
                cols[c] = (pd.Series(v).reindex(indoor.index)
                           if isinstance(v, (list, tuple, pd.Series, np.ndarray))
                           else pd.Series(v, index=indoor.index))
            return pd.DataFrame(cols)

        sp = None if setpoints is None else expand(setpoints, np.nan)
        cool = expand(cooling_flags, True).astype(bool)

        feats = features_from_frames(indoor, cool, sp, sample_interval_s, None)
        index_score = score_features(feats)

        return self._assemble(
            feats, index_score, list(indoor.columns),
            file_id="live_window",
            median_dt_s=sample_interval_s,
            n_rows=int(len(indoor)),
        )

    # -- shared result assembly -------------------------------------------
    def _assemble(self, feats, index_score, all_cars, file_id, median_dt_s, n_rows) -> dict:
        active = list(index_score.index)
        # Cars declared in the headers but reporting nothing cannot be assessed;
        # they go to the tail so the ranking is still a full permutation.
        unassessable = [c for c in all_cars if c not in active]
        ranked = active + unassessable

        margin = (float(index_score.iloc[0] - index_score.iloc[1])
                  if len(index_score) > 1 else float("nan"))

        diagnostics = []
        for r, c in enumerate(ranked, start=1):
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

        top = ranked[0] if ranked else None
        if top is not None and top in active:
            summary = (
                f"Car {top} is the most likely refrigerant-leak location. During active "
                f"cooling it ran {feats.loc[top, 'mean_rel_cooling']:+.2f} degC relative to "
                f"the co-cooling consist median, sat "
                f"{feats.loc[top, 'mean_t_minus_set']:+.2f} degC from its own cooling "
                f"setpoint, and stayed more than {REL_ELEVATION_C:.2f} degC above its peers "
                f"for {feats.loc[top, 'persistence_15m'] * 100:.0f}% of the time on a "
                f"{PERSISTENCE_MINUTES:.0f}-minute rolling basis."
            )
        else:
            summary = "No car reported usable cabin temperature telemetry."

        # Honest confidence: a large gap to rank 2 means the leading diagnosis
        # stands alone; a small gap means the top cars are not distinguishable.
        if not np.isfinite(margin):
            confidence = "SINGLE_CAR"
        elif margin >= 2.0:
            confidence = "CLEAR"
        elif margin >= 0.5:
            confidence = "MODERATE"
        else:
            confidence = "AMBIGUOUS"

        return {
            "file_id": file_id,
            "method": self.METHOD,
            "ranked_cars": "|".join(ranked),
            "ranked_cars_list": ranked,
            "most_likely_faulty_car": top,
            "confidence_margin": round(margin, 3) if np.isfinite(margin) else None,
            "confidence": confidence,
            "consist_size": len(ranked),
            "active_cars": active,
            "unassessable_cars": unassessable,
            "car_diagnostics": diagnostics,
            "diagnostic_summary": summary,
            "sample_interval_s": median_dt_s,
            "n_rows_used": n_rows,
        }

    def predict_from_csv(self, file_source, file_name: str = "acv_data.csv") -> dict:
        """
        Runs ACV thermodynamic leak detection and consist temperature ranking
        from an uploaded CSV or Excel (.xlsx) case file.

        load_case() picks its parser by file extension (pd.read_csv vs
        pd.read_excel), so raw upload bytes are spooled to a temp file with
        the ORIGINAL extension preserved rather than forced through
        pd.read_csv - that would corrupt/reject a binary .xlsx workbook,
        which is the format the real ACV test cases ship in.
        """
        import os
        import tempfile

        suffix = os.path.splitext(file_name)[1] or ".xlsx"
        if isinstance(file_source, (bytes, bytearray)):
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(file_source)
                tmp_path = tmp.name
            try:
                leak_result = self.localise_leak(tmp_path, file_id=file_name)
            finally:
                os.unlink(tmp_path)
        else:
            # Already a path (or an in-memory DataFrame from a caller that
            # parsed it itself) - hand it straight to localise_leak.
            leak_result = self.localise_leak(file_source, file_id=file_name)
        top_car = leak_result.get("most_likely_faulty_car")
        margin = leak_result.get("confidence_margin") or 0.0
        confidence = leak_result.get("confidence", "UNKNOWN")

        if confidence in ("CLEAR", "MODERATE") and top_car:
            status = "ACTION_NEEDED"
            action = "ACTION_REPLACE_FILTER"
            anomaly_score = 0.78
            summary = (
                f"Consist refrigerant evaluation for '{file_name}': Car {top_car} identified "
                f"as the primary thermal anomaly with {confidence} confidence (margin {margin:.2f}). "
                f"Elevated cabin temperature persistence suggests refrigerant pressure loss or duct restriction. "
                f"Recommend servicing AC pack and filter on Car {top_car}."
            )
        else:
            status = "GOOD"
            action = "NONE"
            anomaly_score = 0.15
            summary = (
                f"Consist refrigerant evaluation for '{file_name}': all cars cooling in thermal equilibrium. "
                f"No persistent cabin overheating relative to peer consist median."
            )

        return {
            "subsystem": "acv",
            "file_name": file_name,
            "status": status,
            "verdict": status,
            "anomaly_score": anomaly_score,
            "most_likely_faulty_car": top_car,
            "confidence": confidence,
            "confidence_margin": margin,
            "recommended_action": action,
            "conductor_summary": summary,
            "ranked_cars": leak_result.get("ranked_cars_list", []),
            "car_diagnostics": leak_result.get("car_diagnostics", []),
        }

