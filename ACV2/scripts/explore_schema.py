"""
explore_schema.py
===========================================================================
Probe every ACV case file and report, per canonical signal, whether it is
actually populated and whether it *varies between cars* - the only kind of
signal that can localise a fault to a car.

Run:  python ACV2/scripts/explore_schema.py [--quick]
===========================================================================
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acv2 import config as cfg          # noqa: E402
from acv2.io_loader import discover_cases, load_case, load_labels  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


def numeric(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.apply(pd.to_numeric, errors="coerce")


def probe(panel, faulty: str | None) -> None:
    print("=" * 100)
    print(panel.describe())
    print(f"  schema={panel.meta['schema']}  raw_cols={panel.meta['n_raw_columns']}  "
          f"span={panel.meta['span'][0]} -> {panel.meta['span'][1]}  gaps={panel.meta['n_gaps']}")
    print(f"  ground truth faulty car: {faulty or 'UNKNOWN (test)'}")

    cars = panel.active_cars
    print(f"  header cars={panel.header_cars}  active={cars}  inactive={panel.inactive_cars}")

    # ---- which signals are usable, and do they differ between cars? ------
    rows = []
    for signal in sorted(panel.signals):
        frame = panel.get(signal)[panel.header_cars]
        filled = float(frame.notna().mean().mean())
        num = numeric(frame)
        is_num = num.notna().mean().mean() > 0.5
        if is_num:
            per_car_mean = num.mean()
            between = float(per_car_mean.std())
            within = float(num.std().mean())
            distinct = int(pd.unique(num.values[~np.isnan(num.values)]).size) if num.notna().any().any() else 0
        else:
            codes = frame.astype(str)
            vals = pd.unique(codes.values.ravel())
            distinct = int(len(vals))
            between = within = np.nan
        rows.append({
            "signal": signal, "numeric": is_num, "filled": round(filled, 3),
            "n_distinct": distinct,
            "between_car_sd": None if not is_num else round(between, 4),
            "within_car_sd": None if not is_num else round(within, 4),
        })
    table = pd.DataFrame(rows).sort_values(["numeric", "signal"], ascending=[False, True])
    print("\n  --- signal inventory " + "-" * 60)
    print(table.to_string(index=False))

    # ---- categorical vocabularies ---------------------------------------
    for signal in ("run_mode", "set_mode", "op_mode", "ctrl_mode", "info_valid",
                   "load_halved", "self_check"):
        if signal not in panel.signals:
            continue
        frame = panel.get(signal).astype(str)
        counts = pd.Series(frame.values.ravel()).value_counts()
        print(f"\n  [{signal}] {dict(counts.head(10))}")
        # per-car rate of the leak-relevant categories
        if signal == "run_mode":
            cool = frame.apply(lambda s: s.str.lower().str.contains("cool", na=False)).mean()
            full = frame.apply(lambda s: s.str.lower().str.contains("full cool", na=False)).mean()
            print("    pct_cooling  ", {c: round(100 * cool[c], 1) for c in panel.header_cars if c in cool})
            print("    pct_full_cool", {c: round(100 * full[c], 1) for c in panel.header_cars if c in full})
        if signal == "info_valid":
            inval = frame.apply(lambda s: s.str.lower().str.contains("invalid", na=False)).mean()
            print("    pct_invalid  ", {c: round(100 * inval[c], 2) for c in panel.header_cars if c in inval})

    # ---- the core thermal observable ------------------------------------
    t_in = numeric(panel.get("t_in"))[cars]
    if not t_in.empty:
        run = panel.get("run_mode").astype(str).apply(lambda s: s.str.lower())
        cooling = run.apply(lambda s: s.str.contains("cool", na=False)) if "run_mode" in panel.signals \
            else pd.DataFrame(True, index=t_in.index, columns=cars)
        cooling = cooling.reindex(columns=cars).fillna(False)
        med = t_in.median(axis=1)
        elev = t_in.sub(med, axis=0)
        elev_cool = elev.where(cooling)
        summary = pd.DataFrame({
            "t_in_mean": t_in.mean().round(3),
            "t_in_n": t_in.notna().sum(),
            "n_zero_or_oor": ((t_in <= cfg.CLEANING["t_in_min"]) | (t_in >= cfg.CLEANING["t_in_max"])).sum(),
            "elev_all": elev.mean().round(3),
            "elev_cooling": elev_cool.mean().round(3),
            "elev_p90": elev.quantile(0.90).round(3),
        })
        summary["FAULTY"] = ["<<<" if c == faulty else "" for c in summary.index]
        print("\n  --- thermal observable (deg C, relative to consist median) ---")
        print(summary.to_string())

    # ---- set points ------------------------------------------------------
    for sig in ("t_set_cool", "t_set_heat", "t_out"):
        if sig in panel.signals:
            num = numeric(panel.get(sig))[panel.header_cars]
            print(f"\n  [{sig}] per-car mean: "
                  f"{ {c: (None if np.isnan(num[c].mean()) else round(float(num[c].mean()), 2)) for c in num.columns} }")
            print(f"        global min/max = {np.nanmin(num.values):.2f} / {np.nanmax(num.values):.2f}, "
                  f"distinct={int(pd.unique(num.values[~np.isnan(num.values)]).size)}")

    # ---- refrigerant circuit (rich schema) ------------------------------
    for sig in ("p_low_1", "p_low_2", "p_high_1", "p_high_2"):
        if sig not in panel.signals:
            continue
        num = numeric(panel.get(sig))[cars] if cars else numeric(panel.get(sig))
        desc = pd.DataFrame({
            "mean": num.mean().round(2), "p05": num.quantile(0.05).round(2),
            "p50": num.median().round(2), "p95": num.quantile(0.95).round(2),
            "sd": num.std().round(2), "n": num.notna().sum(),
        })
        desc["FAULTY"] = ["<<<" if c == faulty else "" for c in desc.index]
        print(f"\n  --- {sig} ---")
        print(desc.to_string())

    for sig in ("comp_run_1", "comp_run_2", "cfan_run_1", "comp_fault_1", "comp_fault_2", "sv_1", "sv_2"):
        if sig not in panel.signals:
            continue
        frame = panel.get(sig).astype(str)
        vals = pd.Series(frame.values.ravel()).value_counts().head(6)
        print(f"\n  [{sig}] {dict(vals)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip the 33 MB rich-schema file")
    args = ap.parse_args()

    labels = load_labels()
    files = discover_cases(cfg.TRAIN_DIR) + discover_cases(cfg.TEST_DIR)
    for path in files:
        name = os.path.basename(path)
        if args.quick and os.path.getsize(path) > 10e6:
            print("=" * 100)
            print(f"{name}: skipped (--quick)")
            continue
        panel = load_case(path, verbose=True)
        probe(panel, labels.get(name))


if __name__ == "__main__":
    main()
