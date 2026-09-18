"""
acv2.cleaning
===========================================================================
Turn a raw :class:`~acv2.io_loader.CasePanel` into a physically coherent
:class:`CleanCase`.

Design rule: cleaning only ever removes data that is *physically impossible*
or that the pack itself flags as invalid. It never smooths, imputes or
reshapes a value in a way that could create or hide a temperature difference
between cars, because those differences are the diagnosis.

What the pack self-reports as broken is not thrown away - the rate of
invalid-status flags and sensor dropouts is retained as its own evidence
channel (:mod:`acv2.features`), since a pack that drops out is a pack in
trouble.
===========================================================================
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as cfg
from .io_loader import CasePanel

TRUE_TOKENS = ("1", "1.0", "true", "yes", "on", "running", "open", "closed_no",
               "energized", "valid", "normal")
FALSE_TOKENS = ("0", "0.0", "false", "no", "off", "stopped", "closed", "shut")


# --------------------------------------------------------------------------
# small conversions
# --------------------------------------------------------------------------
def to_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.apply(pd.to_numeric, errors="coerce")


def _lower_text(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.apply(lambda s: s.astype("string").str.strip().str.lower())


def to_boolean(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce a mixed numeric/boolean/text status frame to float 0/1/NaN."""
    num = frame.apply(pd.to_numeric, errors="coerce")
    text = _lower_text(frame)
    out = num.astype("float64")
    for col in frame.columns:
        missing = out[col].isna() & text[col].notna()
        if not missing.any():
            continue
        vals = text.loc[missing, col]
        mapped = pd.Series(np.nan, index=vals.index, dtype="float64")
        mapped[vals.isin(TRUE_TOKENS)] = 1.0
        mapped[vals.isin(FALSE_TOKENS)] = 0.0
        out.loc[missing, col] = mapped
    return out.where(out.isin([0.0, 1.0]))


def contains_token(frame: pd.DataFrame, tokens: tuple[str, ...]) -> pd.DataFrame:
    """Boolean frame: True where the cell text contains any token."""
    text = _lower_text(frame)
    out = pd.DataFrame(False, index=frame.index, columns=frame.columns)
    for token in tokens:
        out |= text.apply(lambda s: s.str.contains(token, na=False, regex=False))
    return out


def _mask_range(frame: pd.DataFrame, lo: float, hi: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (masked frame, out-of-range flags)."""
    num = to_numeric(frame)
    bad = num.notna() & ((num < lo) | (num > hi))
    return num.mask(bad), bad


# --------------------------------------------------------------------------
# clean case
# --------------------------------------------------------------------------
@dataclass
class CleanCase:
    panel: CasePanel
    cars: list[str]                     # cars with usable telemetry
    header_cars: list[str]              # every car in the file headers
    t_in: pd.DataFrame                  # cabin temperature, deg C
    t_set: pd.DataFrame                 # cooling set point, deg C
    t_out: pd.DataFrame                 # per-car outdoor reading, deg C
    ambient: pd.Series                  # consist-level ambient, deg C
    cooling: pd.DataFrame               # bool: pack asked to cool
    demand: pd.DataFrame                # 0 = not cooling, 1 = part, 2 = full
    reporting: pd.DataFrame             # bool: pack transmitting this row
    invalid: pd.DataFrame               # bool: pack flagged its data invalid
    dropout: pd.DataFrame               # bool: implausible cabin temperature
    circuits: dict[int, dict[str, pd.DataFrame]]   # rich schema only
    time: pd.Series
    segment: pd.Series
    report: dict = field(default_factory=dict)

    @property
    def elapsed_days(self) -> pd.Series:
        t0 = self.time.min()
        return (self.time - t0).dt.total_seconds() / 86400.0

    @property
    def has_refrigerant_telemetry(self) -> bool:
        return len(self.circuits) > 0

    def summary(self) -> str:
        r = self.report
        return (f"{self.panel.file_id}: cars={len(self.cars)}/{len(self.header_cars)} "
                f"rows_kept={r['rows_kept']}/{r['rows_total']} "
                f"cooling_rows={r['cooling_rows']} circuits={len(self.circuits)} "
                f"dropouts={r['dropouts_total']} invalid={r['invalid_total']}")

    def subset(self, mask: pd.Series) -> "CleanCase":
        """
        Row subset of the case, used by the block-bootstrap stability check.

        Segment ids are preserved, so rolling and derivative features still
        refuse to cross a discontinuity after resampling.
        """
        mask = pd.Series(mask, index=self.t_in.index).fillna(False).astype(bool)

        def take(obj):
            return obj.loc[mask].reset_index(drop=True)

        return CleanCase(
            panel=self.panel,
            cars=self.cars,
            header_cars=self.header_cars,
            t_in=take(self.t_in), t_set=take(self.t_set), t_out=take(self.t_out),
            ambient=take(self.ambient), cooling=take(self.cooling),
            demand=take(self.demand), reporting=take(self.reporting),
            invalid=take(self.invalid), dropout=take(self.dropout),
            circuits={k: {n: take(v) for n, v in ch.items()} for k, ch in self.circuits.items()},
            time=take(self.time), segment=take(self.segment),
            report=dict(self.report, rows_kept=int(mask.sum()), subset=True),
        )



def clean(panel: CasePanel) -> CleanCase:
    """Apply physical-plausibility cleaning and derive the operating context."""
    cars = panel.active_cars or panel.header_cars
    idx = panel.time.index

    # ---- cabin temperature ---------------------------------------------
    t_in_raw = panel.get("t_in").reindex(columns=panel.header_cars)
    t_in, dropout = _mask_range(t_in_raw, cfg.CLEANING["t_in_min"], cfg.CLEANING["t_in_max"])

    # ---- pack self-reported validity ------------------------------------
    if panel.has("info_valid"):
        invalid = contains_token(panel.get("info_valid").reindex(columns=panel.header_cars),
                                 cfg.INVALID_TOKENS)
    else:
        invalid = pd.DataFrame(False, index=idx, columns=panel.header_cars)
    # A pack that declares its telemetry invalid should not contribute a
    # temperature to the peer comparison.
    run_mode = panel.get("run_mode").reindex(columns=panel.header_cars)
    invalid = invalid | contains_token(run_mode, cfg.INVALID_TOKENS)
    t_in = t_in.mask(invalid)

    # ---- reporting / operating context ---------------------------------
    reporting = run_mode.notna() | t_in_raw.notna()
    cooling = contains_token(run_mode, cfg.COOLING_MODE_TOKENS) & ~invalid
    if not run_mode.notna().any().any():
        # No mode channel at all: assume the pack cools whenever it reports.
        cooling = reporting.copy()
    full_demand = contains_token(run_mode, cfg.FULL_DEMAND_TOKENS)
    demand = pd.DataFrame(0, index=idx, columns=panel.header_cars)
    demand = demand.mask(cooling, 1).mask(cooling & full_demand, 2)

    # ---- set point ------------------------------------------------------
    t_set, _ = _mask_range(panel.get("t_set_cool").reindex(columns=panel.header_cars),
                           cfg.CLEANING["t_set_min"], cfg.CLEANING["t_set_max"])
    # A set point is a step-wise command: holding the last value forward inside
    # a continuous segment is the physically correct reconstruction.
    t_set = t_set.groupby(panel.segment).transform(lambda s: s.ffill().bfill())

    # ---- outdoor / ambient ---------------------------------------------
    t_out, _ = _mask_range(panel.get("t_out").reindex(columns=panel.header_cars),
                           cfg.CLEANING["t_out_min"], cfg.CLEANING["t_out_max"])
    # Ambient air is a property of the train, not of a car: several files only
    # instrument the end cars, so the consist median is the best estimate and
    # it also removes any single-sensor bias from the load proxy.
    ambient = t_out.median(axis=1, skipna=True)
    ambient = ambient.groupby(panel.segment).transform(
        lambda s: s.interpolate(limit_direction="both", limit=30))

    # ---- refrigerant circuits (rich schema) -----------------------------
    circuits: dict[int, dict[str, pd.DataFrame]] = {}
    for k in (1, 2):
        if not (panel.has(f"p_low_{k}") and panel.has(f"p_high_{k}")):
            continue
        p_low, _ = _mask_range(panel.get(f"p_low_{k}").reindex(columns=panel.header_cars),
                               cfg.CLEANING["p_min"], cfg.CLEANING["p_max"])
        p_high, _ = _mask_range(panel.get(f"p_high_{k}").reindex(columns=panel.header_cars),
                                cfg.CLEANING["p_min"], cfg.CLEANING["p_max"])
        comp = to_boolean(panel.get(f"comp_run_{k}").reindex(columns=panel.header_cars))
        fault = to_boolean(panel.get(f"comp_fault_{k}").reindex(columns=panel.header_cars))
        fan = to_boolean(panel.get(f"cfan_run_{k}").reindex(columns=panel.header_cars))
        valve = to_boolean(panel.get(f"sv_{k}").reindex(columns=panel.header_cars))
        # Pressures are only meaningful while the compressor is turning; an
        # idle circuit equalises and says nothing about its charge.
        running = comp.fillna(0.0) > 0.5
        if not running.any().any():          # no compressor channel: use cooling
            running = cooling
        circuits[k] = {
            "p_low": p_low.where(running),
            "p_high": p_high.where(running),
            "p_low_any": p_low,
            "p_high_any": p_high,
            "running": running,
            "comp": comp,
            "fault": fault,
            "fan": fan,
            "valve": valve,
        }

    # ---- drop rows where no car reports anything -----------------------
    keep = reporting[cars].any(axis=1) if cars else reporting.any(axis=1)
    report = {
        "rows_total": int(len(idx)),
        "rows_kept": int(keep.sum()),
        "cooling_rows": int(cooling[cars].any(axis=1).sum()) if cars else 0,
        "dropouts_total": int(dropout.to_numpy().sum()),
        "invalid_total": int(invalid.to_numpy().sum()),
        "dropouts_per_car": {c: int(dropout[c].sum()) for c in panel.header_cars},
        "invalid_per_car": {c: int(invalid[c].sum()) for c in panel.header_cars},
        "reporting_per_car": {c: int(reporting[c].sum()) for c in panel.header_cars},
        "schema": panel.meta.get("schema"),
        "dt_seconds": panel.dt_seconds,
    }

    def sub(frame):
        return frame.loc[keep].reset_index(drop=True)

    circuits_kept = {k: {name: sub(val) for name, val in ch.items()} for k, ch in circuits.items()}

    return CleanCase(
        panel=panel,
        cars=cars,
        header_cars=list(panel.header_cars),
        t_in=sub(t_in),
        t_set=sub(t_set),
        t_out=sub(t_out),
        ambient=sub(ambient.to_frame("a"))["a"],
        cooling=sub(cooling),
        demand=sub(demand),
        reporting=sub(reporting),
        invalid=sub(invalid),
        dropout=sub(dropout),
        circuits=circuits_kept,
        time=sub(panel.time.to_frame("t"))["t"],
        segment=sub(panel.segment.to_frame("s"))["s"],
        report=report,
    )
