"""
acv2.io_loader
===========================================================================
Schema-agnostic loader for ACV case files.

The dataset deliberately ships two different column layouts (8 parameters per
car in five files, 59 in one). Nothing here assumes a fixed parameter list:
the loader reads each file's own headers, splits them into
``Car <NN> - <parameter>``, and maps ``<parameter>`` onto a canonical signal
name via :data:`acv2.config.CANONICAL_SIGNALS`. Unknown parameters are kept
under a slugged name so a future file with new columns still loads.

The output is a :class:`CasePanel`: one wide DataFrame per canonical signal,
indexed by timestamp with the file's own two-digit car identifiers as columns.
That shape makes the peer-comparison physics trivial to express and keeps the
car identifiers byte-identical to the headers, as the submission format
requires.

Reading the 33 MB / 483-column case takes ~1-2 minutes through openpyxl, so
every parse is cached to Parquet keyed on the source file's size and mtime.
===========================================================================
"""
from __future__ import annotations

import glob
import hashlib
import os
import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as cfg

CAR_COL_RE = re.compile(r"^\s*Car\s*([0-9A-Za-z]+)\s*-\s*(.+?)\s*$", re.IGNORECASE)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _slug(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).strip().lower()
    text = re.sub(r"[^0-9a-z]+", "_", text)
    return text.strip("_") or "unnamed"


def _canonical_signal(param: str) -> str:
    """Map a raw per-car parameter name onto a canonical signal name."""
    key = unicodedata.normalize("NFKC", str(param)).strip().lower()
    if key in cfg.CANONICAL_SIGNALS:
        return cfg.CANONICAL_SIGNALS[key]
    # Fall back to substring matching, longest alias first so that specific
    # names win over generic ones.
    for alias in sorted(cfg.CANONICAL_SIGNALS, key=len, reverse=True):
        if alias in key:
            return cfg.CANONICAL_SIGNALS[alias]
    return "x_" + _slug(param)


def _find_time_column(columns) -> str | None:
    for col in columns:
        name = str(col).strip().lower()
        if CAR_COL_RE.match(str(col)):
            continue
        if any(h in name for h in cfg.TIME_COLUMN_HINTS):
            return col
    return None


def _cache_key(path: str) -> str:
    stat = os.stat(path)
    raw = f"{os.path.basename(path)}|{stat.st_size}|{int(stat.st_mtime)}|v3"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# panel
# --------------------------------------------------------------------------
@dataclass
class CasePanel:
    """A single ACV case file, reshaped into per-signal (time x car) frames."""

    file_id: str
    path: str
    time: pd.Series                                    # datetime64, length T
    signals: dict[str, pd.DataFrame]                   # signal -> (T x cars)
    header_cars: list[str]                             # every car in the headers
    dt_seconds: float                                  # median sampling interval
    segment: pd.Series                                 # contiguous-segment id per row
    meta: dict = field(default_factory=dict)

    # -- access -----------------------------------------------------------
    def has(self, signal: str) -> bool:
        return signal in self.signals and self.signals[signal].notna().any().any()

    def get(self, signal: str) -> pd.DataFrame:
        """Wide frame for ``signal``; all-NaN frame if the file lacks it."""
        if signal in self.signals:
            return self.signals[signal]
        return pd.DataFrame(np.nan, index=self.time.index, columns=self.header_cars)

    def series(self, signal: str, car: str) -> pd.Series:
        frame = self.get(signal)
        if car in frame.columns:
            return frame[car]
        return pd.Series(np.nan, index=self.time.index, name=car)

    @property
    def n_rows(self) -> int:
        return len(self.time)

    @property
    def active_cars(self) -> list[str]:
        """Cars with usable cabin-temperature telemetry."""
        t_in = self.get("t_in")
        return [c for c in self.header_cars
                if c in t_in.columns and t_in[c].notna().sum() >= cfg.CLEANING["min_segment_rows"]]

    @property
    def inactive_cars(self) -> list[str]:
        active = set(self.active_cars)
        return [c for c in self.header_cars if c not in active]

    def rows_for_minutes(self, minutes: float) -> int:
        """Convert a physical duration into a row count using this file's dt."""
        dt = self.dt_seconds if self.dt_seconds and self.dt_seconds > 0 else 30.0
        return max(2, int(round(minutes * 60.0 / dt)))

    def describe(self) -> str:
        sig_counts = {k: int(v.notna().any().sum()) for k, v in sorted(self.signals.items())}
        return (f"{self.file_id}: rows={self.n_rows} dt={self.dt_seconds:.0f}s "
                f"cars={len(self.header_cars)} active={len(self.active_cars)} "
                f"segments={self.segment.nunique()}\n  signals={sig_counts}")


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def _read_raw(path: str, cache_dir: str | None, verbose: bool) -> pd.DataFrame:
    """Read the workbook, with a Parquet cache of the raw sheet."""
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(
            cache_dir, f"{os.path.splitext(os.path.basename(path))[0]}.{_cache_key(path)}.parquet")
        if os.path.exists(cache_path):
            if verbose:
                print(f"  [cache] {os.path.basename(cache_path)}")
            return pd.read_parquet(cache_path)
    else:
        cache_path = None

    if verbose:
        size_mb = os.path.getsize(path) / 1e6
        print(f"  [read ] {os.path.basename(path)} ({size_mb:.1f} MB) ...", flush=True)

    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".txt"):
        raw = pd.read_csv(path)
    else:
        raw = pd.read_excel(path)

    if cache_path:
        try:
            # Parquet needs consistent column types; everything not numeric or
            # datetime is stored as string.
            safe = raw.copy()
            safe.columns = [str(c) for c in safe.columns]
            for col in safe.columns:
                if not (pd.api.types.is_numeric_dtype(safe[col])
                        or pd.api.types.is_datetime64_any_dtype(safe[col])):
                    safe[col] = safe[col].astype("string")
            safe.to_parquet(cache_path, index=False)
        except Exception as exc:  # caching is an optimisation, never fatal
            if verbose:
                print(f"  [cache] skipped ({exc})")
    return raw


def load_case(path: str, cache_dir: str | None = cfg.CACHE_DIR, verbose: bool = False) -> CasePanel:
    """Load one ACV case file into a :class:`CasePanel`."""
    raw = _read_raw(path, cache_dir, verbose)
    raw.columns = [str(c) for c in raw.columns]

    # ---- time axis ------------------------------------------------------
    time_col = _find_time_column(raw.columns)
    if time_col is None:
        time = pd.Series(pd.date_range("2000-01-01", periods=len(raw), freq="30s"))
        meta_time = "synthetic"
    else:
        time = pd.to_datetime(raw[time_col], errors="coerce")
        meta_time = time_col
    order = np.argsort(time.values, kind="stable")
    raw = raw.iloc[order].reset_index(drop=True)
    time = time.iloc[order].reset_index(drop=True)

    # ---- split per-car columns -----------------------------------------
    per_signal: dict[str, dict[str, pd.Series]] = {}
    header_cars: list[str] = []
    for col in raw.columns:
        match = CAR_COL_RE.match(col)
        if not match:
            continue
        car_id, param = match.group(1).strip(), match.group(2).strip()
        if _slug(param) in ("model",):          # "Car model" is an id column
            continue
        if car_id not in header_cars:
            header_cars.append(car_id)
        signal = _canonical_signal(param)
        per_signal.setdefault(signal, {})
        series = raw[col]
        if car_id in per_signal[signal]:
            # Two raw columns collapsed onto the same canonical signal (e.g. a
            # file carrying both an outdoor average and a fresh-air probe):
            # keep the one with more data.
            if series.notna().sum() <= per_signal[signal][car_id].notna().sum():
                continue
        per_signal[signal][car_id] = series

    header_cars = sorted(header_cars, key=lambda c: (len(c), c))

    signals: dict[str, pd.DataFrame] = {}
    for signal, by_car in per_signal.items():
        frame = pd.DataFrame({c: by_car.get(c) for c in header_cars})
        frame.index = raw.index
        signals[signal] = frame

    # ---- sampling interval and gap-aware segmentation -------------------
    dt = time.diff().dt.total_seconds()
    dt_median = float(dt.median()) if dt.notna().any() else 30.0
    if not np.isfinite(dt_median) or dt_median <= 0:
        dt_median = 30.0
    gap = dt > cfg.CLEANING["gap_factor"] * dt_median
    segment = gap.fillna(False).cumsum().astype(int)
    segment.name = "segment"

    identifiers = {c: raw[c].dropna().iloc[0] if raw[c].notna().any() else None
                   for c in raw.columns if str(c).strip().lower() in cfg.ID_COLUMNS}

    panel = CasePanel(
        file_id=os.path.basename(path),
        path=os.path.abspath(path),
        time=time,
        signals=signals,
        header_cars=header_cars,
        dt_seconds=dt_median,
        segment=segment,
        meta={
            "time_column": meta_time,
            "n_raw_columns": int(raw.shape[1]),
            "params_per_car": int(len(per_signal)),
            "identifiers": identifiers,
            "schema": "rich" if "p_low_1" in signals else "thin",
            "n_gaps": int(gap.sum()),
            "span": (str(time.min()), str(time.max())),
        },
    )
    if verbose:
        print("  " + panel.describe().replace("\n", "\n  "))
    return panel


def discover_cases(path: str) -> list[str]:
    """Expand a file or directory into a sorted list of case files."""
    if os.path.isfile(path):
        return [os.path.abspath(path)]
    patterns = ("*.xlsx", "*.xls", "*.csv")
    found: list[str] = []
    for pattern in patterns:
        found.extend(glob.glob(os.path.join(path, pattern)))
        found.extend(glob.glob(os.path.join(path, "**", pattern), recursive=True))
    unique = sorted({os.path.abspath(f) for f in found
                     if not os.path.basename(f).startswith("~$")})
    return unique


def load_labels(labels_csv: str = cfg.LABELS_CSV) -> dict[str, str]:
    """``{filename: faulty_car_id}`` from Train_Labels.csv, ids kept as text."""
    if not os.path.exists(labels_csv):
        return {}
    df = pd.read_csv(labels_csv, dtype=str)
    name_col = next(c for c in df.columns if "file" in c.lower())
    car_col = next(c for c in df.columns if "car" in c.lower())
    out = {}
    for _, row in df.iterrows():
        car = str(row[car_col]).strip()
        if car.isdigit():
            car = car.zfill(2)
        out[str(row[name_col]).strip()] = car
    return out
