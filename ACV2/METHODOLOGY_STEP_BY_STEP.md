# ACV2 — Step-by-Step Build Log

A chronological record of how the refrigerant-leak localiser in `ACV2/` was built:
what was inspected, what was decided and why, what broke, and what the evidence was at
each point. Every number quoted here came out of a command run during the build; the
raw outputs are in `ACV2/reports/`.

Companion documents:
- [`README.md`](README.md) — the physics and the final results
- [`reports/evaluation.md`](reports/evaluation.md) — LOOCV, ablation, bootstrap tables
- [`reports/feature_diagnostics.txt`](reports/feature_diagnostics.txt) — per-case feature tables
- [`reports/inspect_test_case.txt`](reports/inspect_test_case.txt) — the test-case deep dive

A comparison against the pre-existing `ACV/` pipeline is in [§18](#18-comparison-with-the-existing-acv-pipeline).

---

## Step 0 — Read the brief and the ground truth

Read `PS3/03_References/ACV/ACV_Subsystem_Info_Kit.md` and listed
`PS3/02_Datasets/ACV/`.

Facts that shaped everything downstream:

| Fact | Consequence |
|---|---|
| 6 training files, exactly one faulty car each | ~6 supervised examples → a classifier will overfit |
| `Train_Labels.csv`: 01→01, 02→02, 03→03, 04→01, 05→04, 06→06 | usable only for validation, not for fitting |
| "**The exact parameter set differs between case files**" | the loader must read headers, never assume columns |
| One file has >60 parameters/car, the rest 8 | two schemas, so potentially two evidence branches |
| `ranked_cars` must use the file's own two-digit ids | car identifiers must survive parsing byte-identical |
| Scoring is linear rank-decay `(n − (r−1))/n` | 2nd place is worth 0.875 — ranking beats classifying |
| 33.8 MB / 483-column file present | parsing cost matters → cache needed |

Also noted the random-permutation baseline for an 8-car file: mean over `r` of
`(n−r+1)/n` = `(n+1)/(2n)` = **0.5625**. Every later result is reported against it.

## Step 1 — Check the environment before writing anything

```powershell
.\venv\Scripts\python.exe ACV2\scripts\_env_check.py   # temporary, deleted afterwards
```

```
3.11.0   pandas 2.3.3   numpy 2.4.6   sklearn 1.9.1   scipy 1.17.1
pyarrow 24.0.0   openpyxl 3.1.5   joblib 1.6.0   matplotlib 3.11.2   statsmodels MISSING
```

Decisions: `pyarrow` present → Parquet cache is free. `statsmodels` missing → write the
regressions with `numpy.linalg.lstsq` rather than add a dependency. `pytest` also turned
out to be absent (found in Step 16) → the test file got a built-in runner.

## Step 2 — Scaffolding: config and a schema-agnostic loader

Wrote `acv2/config.py` and `acv2/io_loader.py` first, so that probing the data used the
same code path as the final model — no throwaway analysis that later diverges from
production.

`config.py` holds the alias table (`"outside temperature sensor reading"`,
`"fresh air temperature detected value"`, `"outdoor average temperature"` → `t_out`),
mode vocabularies, cleaning limits and the feature registry.

`io_loader.py` produces a **`CasePanel`**: one wide `(time × car)` DataFrame per
canonical signal. Design points:

- headers split with `^Car (\w+) - (.+)$`; the parameter is matched against the alias
  table longest-alias-first, and anything unrecognised is kept as `x_<slug>` so a future
  file with new columns still loads;
- `dt_seconds` is the **median** observed sampling interval, and
  `rows_for_minutes(15)` converts a physical window into a row count per file — this is
  the guard against a "15-minute" window silently meaning 5 minutes on the 10 s file;
- a gap larger than 5× the median interval starts a new `segment`; all rolling and
  derivative operations are grouped by it;
- every parse is cached to Parquet keyed on file size + mtime.

## Step 3 — Probe the five thin-schema files

```powershell
.\venv\Scripts\python.exe ACV2\scripts\explore_schema.py --quick
```

The probe reports, per signal: fill rate, whether it is numeric, and crucially whether
it **varies between cars** — a signal that doesn't cannot localise anything.

Thin schema = `t_in`, `t_out`, `t_set_cool`, `t_set_heat`, `run_mode`, `set_mode`,
`info_valid`, `load_halved`, at 30 s.

Findings that changed the design:

1. **The core observable works.** Mean cabin temperature relative to the consist
   median, during cooling:

   | file | faulty | its elevation | next-highest car |
   |---|---|---|---|
   | case_01 | 01 | **+0.457 K** | 03 at +0.080 |
   | case_02 | 02 | **+0.228 K** | 03 at +0.145 |
   | case_03 | 03 | **+0.576 K** | 04 at +0.094 |
   | case_05 | 04 | **+0.181 K** | 02 at +0.160 |
   | case_06 | 06 | **+1.311 K** | 08 at +0.113 |

   A single feature already ranks the faulty car 1st in 5/5 thin files — but the margin
   in case_02 (0.228 vs 0.145) and case_05 (0.181 vs 0.160) is thin enough that one
   channel alone is not a system. This is what motivated the load-stratified,
   persistence, load-sensitivity and progression channels.

2. **`load_halved` is dead.** Every value in every file is `Normal` (or
   `No Load Shedding` in the rich file). It was removed from the feature registry;
   only `full_demand_frac` survived in the `control` group.

3. **`t_out` is a train-level quantity.** In case_05 and case_06 only cars 01 and 08
   carry an outdoor reading. Rather than dropping the load proxy on those files, the
   ambient became the row-wise consist median — which is also the physically correct
   reading of "outdoor air temperature" and removes single-sensor bias.

4. **Set points contain 0.00 dropouts** (`t_set_cool` global min = 0.00 in four files)
   → range-mask 14–32 °C, then forward-fill *within a segment*, because a set point is a
   step-wise command.

5. **An unexpected, very sharp signal.** `ACV Information Valid = Invalid` and
   exact-0.0 °C cabin dropouts occur almost exclusively on the faulty car:

   | file | faulty | its invalid % | any other car |
   |---|---|---|---|
   | case_01 | 01 | 0.36 | 0.00 |
   | case_02 | 02 | 0.48 | 0.00 |
   | case_03 | 03 | 0.01 | 0.00 |
   | case_05 | 04 | 0.00 | 0.00 (channel silent) |
   | case_06 | 06 | 0.00 | 0.00 (channel silent) |
   | **test** | ? | car 04 = **0.21** | 0.00–0.02 |

   High precision, low recall: right 3/3 times it fired, silent twice. Decision: keep it
   as an explicitly **secondary, non-thermodynamic** channel (`integrity_loss`, weight
   0.45 of ~7.1 total), because a pack that flags its own telemetry invalid is
   legitimately a pack in trouble — but never let it outvote the thermodynamics. It is
   reported separately in the README precisely because it disagrees with the thermal
   channels on the test file.

## Step 4 — Probe the rich-schema file

```powershell
.\venv\Scripts\python.exe ACV2\scripts\explore_schema.py     # 244 s, then cached
```

`acv_case_04.xlsx`: 22 262 rows, 483 columns, **10 s** sampling, and only **4 of 8 cars
populated**. Ground truth: car 01.

It carries two independent refrigeration circuits per car with high- and low-side
pressures. First look at the raw per-car medians:

| car | p_low_1 | p_low_2 | p_high_1 | p_high_2 | % full cooling |
|---|---|---|---|---|---|
| **01 (faulty)** | 500 | 680 | 1860 | **1540** | 33.5 |
| 02 | 480 | 480 | 1920 | 1900 | 69.7 |
| 03 | 520 | 500 | 1980 | 1920 | 59.9 |
| 04 | **400** | **360** | **2100** | 2000 | **89.4** |

**This is where the naive answer fails.** "Undercharge ⇒ lowest suction pressure"
points at car 04 — but car 04 is simply working hardest (89% full cooling), and a pack
at full duty legitimately runs a lower suction and a higher head pressure. The
between-car pressure comparison is confounded by duty cycle.

What actually isolates car 01: each pack has **two circuits**, a leak affects **one**,
and the sibling circuit shares the ambient, the cabin *and* the demand command exactly.
Intra-pack differences, taken from the per-circuit medians above (the shipped feature
computes the median of the matched-row difference, which is slightly different but tells
the same story):

| car | Δp_high between its circuits | Δ pressure lift | suction p50 − p05 |
|---|---|---|---|
| **01** | **−320** | **−500** | **400** (680 → 280) |
| 02 | −20 | −20 | 80 |
| 03 | −60 | −40 | 60 |
| 04 | −100 | −60 | 60 |

Car 01 leads on all three by a factor of 3–8×. Design decision: build the refrigerant
branch entirely out of **intra-pack circuit asymmetry** plus **suction excursion**
(intermittent evaporator starvation), not raw between-car pressure levels. Also noted:
`op_mode` is `Invalid` in 87 373 of 87 389 rows and `comp_fault_*` is always 0 — both
useless, so no feature depends on them.

Also checked the thermal branch on this file: car 04 is the warmest (+0.248 K) and car
01 is second (+0.136 K) while actually *meeting* its set point (−0.075 K). So on this
file the thermal branch alone can only reach rank 2 — the refrigerant physics is not a
bonus here, it is the whole diagnosis.

## Step 5 — Cleaning

`acv2/cleaning.py`. Governing rule written into the module docstring: **cleaning only
removes what is physically impossible or what the pack itself flags invalid. It never
smooths or imputes in a way that could create or hide a temperature difference between
cars, because those differences are the diagnosis.**

- cabin temperature masked outside 5–45 °C (this is what the 0.0 °C dropouts are);
- rows the pack flags `Invalid` are excluded from the peer comparison;
- set point range-masked then ffill/bfill **within a segment**;
- ambient = consist median, interpolated within a segment only;
- circuit pressures kept only while that circuit's compressor is running — an idle
  circuit equalises and says nothing about its charge;
- everything removed is **counted per car** and handed to the `integrity_loss` feature
  rather than thrown away;
- `CleanCase.subset(mask)` added later (Step 12) for the bootstrap, preserving segment
  ids so resampling still cannot create a derivative across a gap.

Two bugs caught immediately on first import: `DataFrame.str` does not exist (needed
`.apply(lambda s: s.str...)` per column), and a stray `dtype float` typo in
`to_boolean`. Both fixed before the first successful run.

## Step 6 — The physics module

`acv2/physics.py`. The docstring derives the lumped energy balance and the three
consequences (capacity deficit, load dependence, progression) that justify every
feature. Contents:

- `robust_z` — median/MAD, clipped at ±4, with a documented fallback to σ when the MAD
  is degenerate (quantised channels where most cars are identical);
- `peer_reference` — **leave-one-out** median across cars, so a badly leaking car cannot
  contaminate the reference it is judged against (matters most on the 4-car file);
- `stratified_mean` / `load_bins` — equal-weight averaging over operating strata, the
  defence against the duty-history confound that made raw pressures misleading;
- `binned_slope` — `d(elevation)/d(load)` fitted on bin means, not raw rows, because the
  temperature channels are quantised to 0.25–0.5 K and 9 000 rows are heavily
  autocorrelated;
- `daily_trend` — K/day on 6-hour slot means, so a diurnal cycle cannot look like a trend;
- `identify_thermal_model` — grey-box least squares
  `dT/dt = a(T_out − T_in) + b·u + c`, returning `UA/C`, cooling authority `Q_max/C`
  and **R², so a non-identification is reported rather than hidden**;
- `pulldown_rate` — 5th percentile of sustained falling rates, with isolated negative
  steps rejected as quantisation noise;
- `cusum_onset` — one-sided CUSUM on the peer-relative residual, giving a leak-onset day;
- `circuit_indicators` — lift, ratio, suction excursion and variability per circuit.

## Step 7 — Features

`acv2/features.py`, two branches, 20 declared features. Deliberate choices:

- elevation is computed from cabin temperatures **masked to cooling rows only**, so a
  sibling that happens to be ventilating (and therefore warm) never pollutes the
  reference;
- the load proxy is `ambient − median(t_set)`, i.e. the cooling span the packs are asked
  to deliver;
- the refrigerant features are computed on rows where **both** circuits of that car are
  running, making the intra-pack comparison strictly like-for-like;
- asymmetries are normalised by the pack's own mean (dimensionless), so the indicator
  does not depend on pressure units — which the workbook never states;
- every declared feature is guaranteed to exist as a column on every schema, so no
  downstream code has to test for schema-dependent columns.

## Step 8 — Detectors and ranker

`acv2/detectors.py`: oriented robust z → weighted pool → IsolationForest second opinion.

Three decisions worth recording:

1. **Weights are renormalised over the features the file actually supports.** A
   thin-schema file is scored on thermal evidence, not penalised for absent pressure
   channels.
2. **The IsolationForest is signed by the physics projection** and floored at zero, so a
   car that is anomalously *good* can never be promoted by "outlierness".
3. `informative_features` drops any channel that is constant or has <2 values in this
   file, which is how dead channels (`load_halved`, unidentified grey-box fits) fall out
   automatically instead of injecting noise.

`acv2/ranker.py` assembles the verdict and enforces the submission contract: cars with
no usable telemetry (four of eight in the rich file) are appended **after** every
diagnosable car in header order — never interleaved, never dropped. `load_model()` falls
back to the physical prior when no artefact exists, so inference never depends on a
pickle.

## Step 9 — First full run, and one real bug

```powershell
.\venv\Scripts\python.exe -W ignore ACV2\scripts\diagnose_features.py
```

Result: **6/6 rank 1, mean rank-decay score 1.0000** with the uncalibrated physical
prior — including `acv_case_04`, where the refrigerant branch lifts car 01 from 2nd to
1st (score 0.984 vs 0.723 for car 04).

But two channels were silently missing from the score:

- **`capacity_shortfall` was degenerate.** Its median was exactly `0.0` for every car in
  case_01 and case_04, because `T_in` and `T_set` are quantised to the same grid and a
  healthy pack sits exactly on set point most of the time. Verified with a throwaway
  probe (`_debug_shortfall.py`, since deleted):
  `car 01: shortfall n=4310 median=0.0` for all cars.
  **Fix:** use the **mean**, which keeps the integral of the deficit. The channel then
  became computable on all 6 files and scored 0.979 on its own.
- **`greybox_cool_rate` identifies only on case_06** (R² ≈ 0.001 elsewhere). Cause: the
  packs cool 85–99% of the time, so the cooling indicator `u` has too little variation
  to separate `UA/C` from `Q_max/C`. This was left as-is: the function reports
  non-identification and the feature drops out, which is the honest behaviour. It is
  listed as a caveat in the README rather than papered over.

Single-channel discrimination (rank-decay score of each channel used alone, six files):

| channel | mean score | files | top-1 |
|---|---|---|---|
| `elev_mean`, `elev_load_stratified`, `elev_persistence`, `setpoint_error` | 0.979 | 6 | 5 |
| `capacity_shortfall` | 0.979 | 6 | 5 |
| `cusum_fraction` | 0.938 | 6 | 4 |
| `elev_p90` | 0.917 | 6 | 4 |
| `load_sensitivity` | 0.875 | 6 | 3 |
| `pulldown_rate` | 0.750 | 6 | 3 |
| `elev_trend` | 0.667 | 6 | 3 |
| `full_demand_frac` | **0.531** | 4 | 0 |
| `integrity_loss` | 1.000 | 3 | 3 |
| refrigerant channels | 0.625–1.000 | 1 | 5 of 7 |

`full_demand_frac` scores **below** the 0.5625 random baseline. It was kept (physically
motivated, 4% of the weight, ablation delta 0.000) but is flagged as the weakest channel
in the README rather than quietly left to look like a contributor.

## Step 10 — Evaluation machinery

`acv2/evaluate.py` + `scripts/evaluate_loocv.py`:

- `rank_decay_score` implementing the competition metric exactly;
- `CaseSet`, which loads/cleans/extracts each file **once** and re-scores it under
  different weights — this is what makes LOOCV and ablation cheap;
- `calibrate_groups` — coordinate ascent over six group multipliers on a coarse grid
  `{0, 0.5, 1, 1.5, 2}`, ties broken toward the physical prior;
- `loocv` — re-runs calibration from scratch on the other five files for each held-out
  file, so the held-out score never sees its own tuning;
- `ablation` — each group zeroed, and each group alone;
- `block_bootstrap_stability` (added in Step 12).

`pandas.to_markdown` needed the absent `tabulate`, so `acv2/reporting.py` was written
instead of adding a dependency (with `|` escaped inside cells, since it is the
`ranked_cars` separator).

## Step 11 — Cross validation and ablation

```powershell
.\venv\Scripts\python.exe -W ignore ACV2\scripts\evaluate_loocv.py
```

```
[1] physical prior, no calibration      mean score = 1.0000   (random baseline 0.5625)
[2] leave-one-file-out                  LOOCV mean score = 1.0000
```

Calibration never found anything to change — the prior already saturates the metric —
so the shipped weights *are* the physics, which is the strongest available statement
against six-file overfitting.

Ablation, which is the more informative table:

| removed | score | Δ |
|---|---|---|
| (nothing) | 1.000 | — |
| thermal / capacity / progression / control / integrity | 1.000 | 0.000 |
| refrigerant | 0.979 | −0.021 |
| all but thermal | 0.979 | −0.021 |
| all but capacity | 0.958 | −0.042 |
| all but integrity | 0.958 | −0.042 |
| all but progression | 0.792 | −0.208 |
| all but control | 0.646 | −0.354 |

Reading: **no single branch is load-bearing** — zeroing any one group keeps ≥0.979 — and
three independent branches each nearly solve the task on their own. The only −0.021 is
the refrigerant group, which is exactly `acv_case_04` falling to rank 2 as predicted in
Step 4.

## Step 12 — Stability, because six files is not many

Added `CleanCase.subset()` and `block_bootstrap_stability`: re-rank each case on random
70% subsets of its **6-hour blocks**. Whole blocks, not rows — consecutive 30 s samples
are strongly autocorrelated, and an i.i.d. bootstrap would grossly overstate the
effective sample size.

Top-1 frequency for the true faulty car over 40 draws: case_01 1.00, case_02 1.00,
case_03 1.00, case_04 0.80, case_05 0.80, case_06 1.00.

## Step 13 — Train and export

```powershell
.\venv\Scripts\python.exe -W ignore ACV2\scripts\train.py
```

Writes `artifacts/acv2_model.joblib` (weights, orientations, provenance, measured
scores), `artifacts/train_features.csv` (every number behind every verdict) and
`artifacts/training_summary.json`. **No estimator is pickled** — there is nothing to fit,
so there is no version-fragile object to unpickle later.

## Step 14 — Inference on the held-out case

```powershell
.\venv\Scripts\python.exe -W ignore ACV2\predict.py -i PS3\02_Datasets\ACV\Test -o ACV2\acv_predictions.csv --report
```

```
acv_test_case.xlsx → 01|04|03|08|07|06|02|05
```

## Step 15 — Interrogate that verdict before trusting it

The test case's thermal margin is far smaller than any training case (+0.11 K vs
0.23–1.31 K), so `scripts/inspect_case.py` was written to stress it: contributions,
daily elevation profile, bootstrap, and a sensitivity sweep that re-ranks with each
group zeroed.

```
top-1 frequency: 01 = 1.00   (40/40 draws)

without (nothing)   -> 01|04|03|08|07|06|02|05
without thermal     -> 01|04|03|08|07|06|02|05
without capacity    -> 01|04|03|07|08|06|02|05
without progression -> 01|04|03|08|07|06|02|05
without control     -> 01|04|03|08|07|06|02|05
without refrigerant -> 01|04|03|08|07|06|02|05
without integrity   -> 01|03|04|08|07|06|02|05
```

Car 01 is first under **every** ablation and in every bootstrap draw. The genuine
disagreement is about 2nd place: thermal/capacity/progression → 01, while
`integrity_loss` → 04 (39 invalid/dropout events versus 0–4 for every other car).
CUSUM separates exactly those two from the rest (peaks 23.9 and 15.2 versus 0.7–3.3).

Decision recorded in the README: keep both in the top two — under rank-decay the
expected value of 01-then-04 versus the reverse is nearly identical (1.000/0.875 either
way) — and let the thermodynamics, the primary evidence of lost cooling capacity, set
the order.

## Step 16 — Verification suite, and a design fix it forced

Wrote `tests/test_acv2.py`: 21 checks over the metric (against the worked example in the
problem statement), schema agnosticism, cleaning, physics signs, and the output
contract. `pytest` is not installed in this venv, so the file also carries a built-in
runner.

First run: **20 passed, 1 failed**. The failure was substantive, not cosmetic:

```
FAIL test_cusum_detects_a_sustained_step_and_ignores_a_spike
```

The CUSUM docstring claimed a large isolated excursion would not alarm. It did: five
samples at 8 K accumulate `5 × (8 − 0.15) = 39 ≫ h = 6`. The claim was false, and a
sensor glitch tripping the leak-onset detector is a real defect, not a test artefact.

**Fix:** cap each step at `cusum_clip_c = 1.0` K before accumulating, so the statistic
responds to *persistence* rather than magnitude. The spike now accumulates 4.25 and
stays below threshold; a sustained +0.8 K still alarms after ~10 slots (≈2.5 h). Re-ran
everything afterwards: **21/21 passing**, and train/LOOCV/prediction all unchanged.

## Step 17 — Documentation, cleanup, final pass

Deleted the temporary probes (`_env_check.py`, `_debug_shortfall.py`, the raw `_probe*`
and `_diag*` dumps), converted the retained reports from PowerShell's UTF-16 redirect
output to UTF-8, and wrote `README.md` (physics, results, caveats, layout, commands)
plus `requirements.txt`.

Final verification, all five stages in one pass:

| stage | result |
|---|---|
| tests | 21 passed, 0 failed |
| train | prior 1.0000, LOOCV 1.0000, 3 artefacts written |
| evaluate | 1.0000; bootstrap 0.8–1.0; `reports/evaluation.md` written |
| predict | `acv_test_case.xlsx,01\|04\|03\|08\|07\|06\|02\|05` |
| format | header and shape identical to `PS3/04_Example_Submission/acv_predictions.csv` |

---

## 18. Comparison with the existing `ACV/` pipeline

`ACV/` already held a working solution, which I read before starting (`acv_pipeline.py`,
`_audit_verify.txt`, `_audit_profile.txt`) and used to shortcut data discovery. `ACV2/`
is an independent rebuild, not a refactor.

One caveat on reading the table below: `ACV/` contains two variants — the shipped
`acv_pipeline.ACVPredictor` and a later audited "fixed ranker" whose results are recorded
in `ACV/_audit_verify.txt`. Where a row describes a defect that the audited version
already corrected, that is stated explicitly. The rows are scoped to what I actually read
in those files; I did not re-run `ACV/` as part of this build.

| | `ACV/` (existing) | `ACV2/` (this build) |
|---|---|---|
| Reported score on the 6 training cases | 0.9583 shipped; 0.9792 for the audited fixed ranker | **1.0000**, and 1.0000 under LOOCV |
| `acv_case_04` | rank 2 (0.875) | **rank 1** |
| Features | 6, all thermal | 20 across 6 evidence groups |
| Rich-schema pressures | **discarded** — `clean_telemetry` renames the 483-column file's columns onto the thin-schema names, so the two refrigeration circuits are never read | dedicated refrigerant branch; intra-pack circuit asymmetry + suction excursion |
| Fitted component | pickled `LogisticRegression` pairwise model, blended 0.75/0.25 | **none** — no estimator is pickled |
| Runs without an artefact | `acv_pipeline.ACVPredictor.__init__` raises `FileNotFoundError` without `acv_model_bundle.joblib` (the later audited ranker removed this dependency) | yes — falls back to the physical prior, which is what the scores were measured with |
| Peer reference | median including the car itself | leave-one-out median |
| z-scores | mean/σ (inflated by the outlier being sought) | median/MAD (immune to it) |
| Persistence window | `window=30` rows hardcoded in `acv_pipeline.py` → 15 min at 30 s but **5 min** at 10 s (corrected in the audited ranker) | `rows_for_minutes(15)` per file: 30 rows and 90 rows respectively |
| Gap handling | rolling means run across data gaps in `acv_pipeline.py` (segmented in the audited ranker) | rolling/derivative operations segmented at >5× median `dt`, everywhere |
| Duty-cycle confound | not addressed | load-stratified and demand-matched statistics |
| Leak vs sensor bias | not distinguished | `load_sensitivity` slope, by construction |
| Onset time | not produced | CUSUM change point (test case: day 1.9) |
| Robustness evidence | accuracy audit | ablation + block bootstrap + per-channel discrimination |
| Known defect | `sklearn` 1.6.1 pickle loaded under 1.9.1 raises `InconsistentVersionWarning` | not applicable — nothing is unpickled |
| Tests | audit scripts | 21 automated checks incl. the metric and the output contract |

**The two pipelines agree on the test case.** `ACV/`'s `predict.py` produced
`01|04|03|08|07|06|02|05` (recorded in `ACV/_audit_verify.txt`); `ACV2/` produced the
same string from independent code, different features, different statistics and an extra
evidence branch. That convergence is worth more than either result alone — though it is
partly non-independent, since both rest on the same peer-relative cabin-temperature
observable, which is the only leak signature the thin schema exposes.

Where `ACV2/` is genuinely better: `acv_case_04` (real refrigerant physics rather than a
thermal proxy), no pickled estimator to rot, correct physical windows and gap handling,
and the ability to say *why* a car was picked and *how confident* to be. Where it is not
better: it cannot escape the same six labelled files, and its refrigerant branch is
validated on exactly one of them.
