"""
Temporary diagnostic: is case_04's rank-2 miss a DATA CLEANING problem?

Tests, in order of increasing suspicion:
  1. How much data does cleaning actually touch in case_04?
  2. Ablation: no cleaning at all vs current cleaning -> does Car 01's rank move?
  3. Is the 4-car peer group itself the problem (median of 4)?
  4. Regime mismatch: Car 01 sits in Half Cooling 58.7% of the time while its
     peers sit in Full Cooling. Comparing them is not like-for-like. Does a
     regime-matched comparison surface Car 01?
  5. Is Car 04 (which we rank 1st) simply a hotter-running but healthy unit?
"""
import os
import numpy as np
import pandas as pd

from acv_physics_ranker import (
    load_case, cooling_mask, setpoints, robust_z,
    FEATURE_ORDER, PHYSICS_WEIGHTS, TEMP_MIN_C, TEMP_MAX_C, REL_ELEVATION_C,
)

pd.set_option("display.width", 250)
HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "..", "PS3", "02_Datasets", "ACV", "Train", "acv_case_04.xlsx")
TRUE = "01"
RULE = "=" * 95

df, colmap, all_ids, tcol = load_case(P)
cars = [c for c in all_ids if (c, "indoor") in colmap]
raw = pd.DataFrame({c: pd.to_numeric(df[colmap[(c, "indoor")]], errors="coerce") for c in cars})
cars = [c for c in cars if raw[c].notna().any()]
raw = raw[cars]
mode = pd.DataFrame({c: df[colmap[(c, "run_mode")]].astype(str) for c in cars})
sp = pd.DataFrame({c: pd.to_numeric(df[colmap[(c, "set_cool")]], errors="coerce") for c in cars})

print(RULE)
print("1. WHAT DOES CLEANING ACTUALLY TOUCH IN case_04?")
print(RULE)
print(f"  rows={len(df)}  cars reporting temperature={cars}")
print(f"  validity flag column present: {any((c, 'valid') in colmap for c in cars)}")
oor = ((raw <= TEMP_MIN_C) | (raw >= TEMP_MAX_C))
print(f"  readings outside {TEMP_MIN_C}-{TEMP_MAX_C} C, per car:")
print("   ", oor.sum().to_dict())
print(f"  NaN readings per car: {raw.isna().sum().to_dict()}")
print(f"  exact 0.0 readings   : {(raw == 0.0).sum().to_dict()}")
print(f"  temp range per car   :")
print(raw.describe().loc[["min", "25%", "50%", "75%", "max"]].to_string())

cooling = mode.apply(lambda s: s.str.contains("Cooling", case=False, na=False))
print(f"\n  out-of-range readings that survive the cooling-mode filter:")
print("   ", (oor & cooling).sum().to_dict())


def build_features(indoor, cool, setp, min_peers=3):
    ic = indoor.where(cool)
    enough = ic.notna().sum(axis=1) >= min(min_peers, ic.shape[1])
    rel = ic.sub(ic.median(axis=1), axis=0).where(enough)
    tms = ic.sub(setp).where(enough)
    n = rel.notna().sum().replace(0, np.nan)
    return pd.DataFrame({
        "mean_rel_cooling": rel.mean(),
        "mean_t_minus_set": tms.mean(),
        "p95_rel": rel.quantile(0.95),
        "frac_rel_elevated": (rel > REL_ELEVATION_C).sum() / n,
        "persistence_15m": (rel.rolling(90, min_periods=30).mean() > REL_ELEVATION_C).sum() / n,
    }).fillna(0.0)


def rank_of(feats, true=TRUE, weights=PHYSICS_WEIGHTS):
    z = pd.DataFrame({c: robust_z(feats[c]) for c in FEATURE_ORDER}, index=feats.index)
    s = sum(weights[c] * z[c] for c in FEATURE_ORDER).sort_values(ascending=False)
    order = s.index.tolist()
    r = order.index(true) + 1
    n = 8  # all 8 header cars must be ranked
    return r, (n - (r - 1)) / n, "|".join(order), s


print()
print(RULE)
print("2. ABLATION -- does cleaning change the answer at all?")
print(RULE)
variants = {
    "no cleaning (raw temps)":        raw.copy(),
    "range mask only (current)":      raw.mask(oor),
    "range mask + drop all-NaN rows": raw.mask(oor).dropna(how="all"),
}
for label, ind in variants.items():
    cm = cooling.loc[ind.index]
    f = build_features(ind, cm, sp.loc[ind.index])
    r, sc, order, _ = rank_of(f)
    print(f"  {label:<32} Car {TRUE} rank={r}  score={sc:.3f}  order={order}")

print()
print(RULE)
print("3. IS THE 4-CAR PEER GROUP THE PROBLEM? (leave-one-peer-out)")
print(RULE)
base = raw.mask(oor)
for drop in cars:
    if drop == TRUE:
        continue
    sub = [c for c in cars if c != drop]
    f = build_features(base[sub], cooling[sub], sp[sub], min_peers=3)
    z = pd.DataFrame({c: robust_z(f[c]) for c in FEATURE_ORDER}, index=f.index)
    s = sum(PHYSICS_WEIGHTS[c] * z[c] for c in FEATURE_ORDER).sort_values(ascending=False)
    print(f"  excluding Car {drop}: order={'|'.join(s.index)}   "
          f"Car {TRUE} rank={s.index.tolist().index(TRUE) + 1} of {len(sub)}")

print()
print(RULE)
print("4. REGIME MISMATCH -- Car 01 runs Half Cooling, peers run Full Cooling")
print(RULE)
for c in cars:
    vc = mode[c].value_counts()
    tot = len(mode)
    print(f"  Car {c}: " + ", ".join(f"{k}={v/tot*100:.1f}%" for k, v in vc.head(5).items()))

full = mode.apply(lambda s: s.str.contains("Full Cooling", na=False))
print("\n  (a) restrict to timestamps where ALL cars are in FULL cooling:")
all_full = full.all(axis=1)
print(f"      usable timestamps = {int(all_full.sum())} of {len(df)}")
if all_full.sum() > 100:
    f = build_features(base[all_full], full[all_full], sp[all_full])
    r, sc, order, s = rank_of(f)
    print(f"      Car {TRUE} rank={r} score={sc:.3f} order={order}")
    print(f"      mean_rel: {f['mean_rel_cooling'].round(3).to_dict()}")

print("\n  (b) 'cannot reach full capacity' = fraction of cooling time in HALF cooling")
half = mode.apply(lambda s: s.str.contains("Half Cooling", na=False))
half_frac = (half & cooling).sum() / cooling.sum()
print(f"      {half_frac.round(3).to_dict()}   <-- Car {TRUE} is the true fault")
print("      ranking by this feature alone: "
      + "|".join(half_frac.sort_values(ascending=False).index))

print()
print(RULE)
print("5. IS CAR 04 A HOTTER-RUNNING HEALTHY UNIT, OR A SECOND SUSPECT?")
print(RULE)
ic = base.where(cooling)
med = ic.median(axis=1)
out = pd.DataFrame({
    "mean_indoor_cooling": ic.mean(),
    "mean_setpoint": sp.where(cooling).mean(),
    "mean_T_minus_set": ic.sub(sp).mean(),
    "mean_rel_to_peers": ic.sub(med, axis=0).mean(),
    "half_cool_frac": half_frac,
})
print(out.round(3).to_string())
print(f"\n  Car 04 setpoint is {'LOWER' if out.loc['04','mean_setpoint'] < out['mean_setpoint'].drop('04').mean() else 'HIGHER'} "
      f"than the others' average ({out.loc['04','mean_setpoint']:.2f} vs "
      f"{out['mean_setpoint'].drop('04').mean():.2f})")
