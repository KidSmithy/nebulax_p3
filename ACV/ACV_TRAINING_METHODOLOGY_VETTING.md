# ACV Refrigerant Leakage Localisation — Method, Implementation & Honest Results

**Subsystem**: Train Air Conditioning & Ventilation (ACV)
**Task**: locate the single refrigerant-leaking car in an 8-car consist, and rank every car from most to least likely faulty
**Primary artefacts**: [`acv_physics_ranker.py`](./acv_physics_ranker.py) (method), [`predict.py`](./predict.py) (submission CLI), [`validate_physics_ranker.py`](./validate_physics_ranker.py) (checks)
**Model artefact**: none. The ranker has no fitted parameters and loads no pickle.

> **This document supersedes the earlier version of this file.** The previous
> version reported a cross-validation table whose ground-truth column did not
> match `Train_Labels.csv`, and a test-case diagnostic table whose feature
> values were not reproducible. Section 7 lists every correction.

---

## 1. Task and scoring

Each case file is one journey of continuous telemetry from all cars of one
train, sampled every 30 s (10 s for `acv_case_04`). Exactly one car has a
refrigerant leak. The deliverable is a full ranking, scored by **linear rank
decay**:

$$\text{score} = \frac{n - (r - 1)}{n}$$

with `r` the rank given to the true faulty car out of `n` ranked. Top-1 scores
1.000; rank 2 on an 8-car consist still scores 0.875. So the objective is to
put the most likely car first **and** to order the remainder sensibly — a
ranking problem, not a classification problem.

---

## 2. Method choice: physics-informed cross-car ranking, not anomaly detection

Two candidate approaches were considered. The physics ranker was chosen.

| | Physics-informed cross-car index | Unsupervised anomaly detection |
|---|---|---|
| Knows *which direction* is faulty | Yes — a leak reduces cooling capacity, so the cabin runs **warm** | No. An anomaly score flags "different", so the coldest or the most erratic car scores just as high as the warmest |
| Cancels weather / solar / speed | Yes — the peer cars are the control group, measured at the same instant | No. A per-car model trained on its own history drifts with season and route |
| Needs labelled data | No | No, but needs a clean "normal" reference period that the data does not delimit |
| Validatable on 6 journeys | Yes, and it was never fitted, so the check is not circular | No — nothing to hold out meaningfully |
| Output a depot can act on | "Car 01 ran +0.11 °C above its peers for 37% of cooling time and never reached its own setpoint" | "Car 01 anomaly score 0.83" |

Three pieces of evidence from this repository's own prior experiments settled it:

1. **Direction matters more than sophistication.** Systematically negating each
   feature's sign swings the mean score from 0.90 down to 0.40
   (`_audit_combo.txt`, section 2). Physics supplies the sign for free; an
   unsupervised detector must guess it.
2. **The learned component contributed exactly nothing.** The previous design
   blended the physics index with an L2 logistic pairwise ranker. Physics-only
   scored 0.9583 and the blend scored 0.9583 — identical
   (`_audit_loocv.txt`, Part A). The model was carrying no information.
3. **Searching for a better feature set made things worse.** Feature-combination
   selection performed *inside* each cross-validation fold scored **0.8125**,
   against 0.9583 for a single physics feature used with no selection at all
   (`_audit_combo.txt`, section 4). With 6 journeys, fitting buys variance.

**Consequence for the implementation:** the pairwise logistic regression, the
0.75/0.25 ensemble and the `acv_model_bundle.joblib` pickle were all removed.
What remains is one deterministic closed-form index.

---

## 3. The physical argument

A leak reduces the refrigerant mass circulating through the evaporator. Cooling
capacity falls, so the affected cabin drifts above its setpoint and above its
sibling cars. Every car on the consist shares the same outdoor weather, solar
load, train speed and station door cycles at every instant — which is exactly
what licenses the cross-car comparison. The median cabin temperature of the
cars that are **cooling at that same timestamp** is therefore a live
environmental reference, and subtracting it removes the dominant nuisance
variable with no fitted parameters:

$$\Delta T_{\text{rel}}(c,t) = T_{\text{in}}(c,t) - \underset{k \,\in\, \text{co-cooling}(t)}{\operatorname{median}}\ T_{\text{in}}(k,t)$$

$$\Delta T_{\text{set}}(c,t) = T_{\text{in}}(c,t) - T_{\text{setpoint,cool}}(c,t)$$

`ΔT_rel` is peer-relative and immune to a consist-wide hot day. `ΔT_set` is
absolute and survives even if several cars degrade together. Using both means
neither failure mode is blind.

---

## 4. Pipeline

```
[case .xlsx]
   │  header inspected first; only the ~40 needed columns are read
   │  (acv_case_04 is 33 MB / 483 columns)
   ▼
[1] Schema harmonisation
      Indoor  <- 'Indoor Average Temperature' | 'Passenger Cabin Temperature Detected Value'
      Setpt   <- 'ACV Control Temperature (Cooling)' | 'Target Temperature Value'
      Outdoor <- 'Outdoor Average Temperature' | 'Outside Temperature Sensor Reading'
                 | 'Fresh Air Temperature Detected Value'
      Car ids are taken from the file's own headers, never assumed.
   ▼
[2] Cleaning — mask to NaN, never fill
      - rows the ACV flags 'Invalid' via 'ACV Information Valid'
      - readings outside 5–45 °C (CAN-bus dropouts arrive as 0.0 °C)
      - drop rows where no car reports at all (depot / power-off)
   ▼
[3] Regime gating
      A car counts as cooling when its running mode contains 'Cooling'
      ('Automatic Cooling', 'Full Cooling', 'Half Cooling').
      The peer median uses ONLY cars cooling at that same timestamp, and a
      timestamp is used only if >= 3 cars are co-cooling.
   ▼
[4] Five same-signed statistics, higher = more suspect
      mean_rel_cooling   mean ΔT_rel                                   weight 1.00
      mean_t_minus_set   mean ΔT_set                                   weight 0.50
      p95_rel            95th percentile of ΔT_rel (peak-load tail)    weight 0.25
      frac_rel_elevated  fraction of cooling time ΔT_rel > 0.20 °C     weight 0.25
      persistence_15m    fraction of time a 15-min rolling mean of
                         ΔT_rel stays > 0.20 °C                        weight 0.25
   ▼
[5] Robust within-consist standardisation (median / 1.4826·MAD), weighted sum
   ▼
[6] Descending sort; cars present in the headers but reporting no temperature
    are appended to the tail, so the output is always a full permutation
```

Two implementation points that materially affect correctness:

- **No forward-filling.** Journeys span several days with overnight gaps.
  Interpolating across them fabricates cooling-window data. Every statistic is
  NaN-aware, so filling buys nothing.
- **The 15-minute window is measured in physical time and restarted at every
  acquisition gap.** The window is `round(15·60 / median_dt)` rows — 30 rows at
  30 s sampling, 90 rows at `acv_case_04`'s 10 s — and the rolling mean is
  computed per contiguous segment, split wherever the gap exceeds 3·median_dt.
  A fixed 30-row window would silently mean 5 minutes on case 04, and a
  non-segmented rolling mean would let a 15-minute window straddle a 9-hour
  depot shutdown.

### Why median/MAD rather than mean/σ

The car being searched for is an outlier among eight. It inflates σ and drags
the mean toward itself, shrinking its own z-score — the estimator is
contaminated by the signal. Median and MAD are unaffected by one outlier in
eight. Measured effect on the test case: the gap between rank 1 and rank 2
widens from 3.27 to 4.66 index points, with the same car first (§6, B3).

### What is deliberately *not* used

`acv_case_04` also carries per-circuit refrigeration pressures, compressor and
fan states — 63 parameters per car. **None of it is used**, because the test
file and the other five training files carry only the 8 standard SCADA
parameters. Training on sensors that will not exist at inference time would
break train/inference parity. §6 shows exactly what this costs.

---

## 5. Results on the labelled journeys

Ground truth is `PS3/02_Datasets/ACV/Train_Labels.csv`. Reproduce with
`python validate_physics_ranker.py`.

| Case | n ranked | assessable | dt | True car | Rank | Score | Margin | Ranking produced |
|---|---|---|---|---|---|---|---|---|
| case_01 | 8 | 8 | 30 s | 01 | **1** | 1.000 | 20.07 | `01\|03\|04\|02\|07\|08\|05\|06` |
| case_02 | 8 | 8 | 30 s | 02 | **1** | 1.000 | 3.35 | `02\|03\|07\|08\|06\|04\|01\|05` |
| case_03 | 8 | 8 | 30 s | 03 | **1** | 1.000 | 7.92 | `03\|02\|04\|01\|08\|07\|05\|06` |
| case_04 | 8 | 4 | 10 s | 01 | 2 | 0.875 | 2.99 | `04\|01\|02\|03\|05\|06\|07\|08` |
| case_05 | 8 | 8 | 30 s | 04 | **1** | 1.000 | 0.85 | `04\|02\|01\|07\|06\|03\|08\|05` |
| case_06 | 8 | 8 | 30 s | 06 | **1** | 1.000 | 20.77 | `06\|08\|04\|02\|05\|03\|01\|07` |

**Mean linear rank-decay = 0.9792. Top-1 on 5 of 6. Random-permutation
baseline = 0.5625.**

These six numbers are a sanity check, not a generalisation estimate. Nothing
was fitted, so the check is not circular — but six journeys cannot establish a
confidence interval, and it would be wrong to quote 0.9792 as expected
held-out performance.

### The case_04 miss, explained

Case 04 is a different rolling-stock generation: 483 columns, 10 s sampling,
only 4 cars populated, and running modes `Full Cooling` / `Half Cooling`
instead of `Automatic Cooling`. The thermal index ranks Car 04 first and the
true faulty Car 01 second, because **Car 01 is not the hottest car** — its
cabin sits 0.07 °C *below* its own setpoint on average.

Reading case_04's richer sensors directly (verified in this session) shows why:

| Car | Sys-1 high | Sys-2 high | Asymmetry | Pressure ratio | Full cool | Half cool | T−setpoint |
|---|---|---|---|---|---|---|---|
| **01** (true fault) | 1844.1 | 1601.5 | **242.6** | 3.64 / 3.62 | 33.5% | **58.7%** | −0.07 |
| 02 | 1835.7 | 1813.6 | 22.1 | 3.67 / 3.64 | 69.7% | 25.4% | −0.18 |
| 03 | 1834.5 | 1798.9 | 35.6 | 3.54 / 3.47 | 59.9% | 34.9% | −0.17 |
| 04 (ranked 1st) | 2069.2 | 1968.0 | 101.2 | **5.16 / 5.41** | **89.4%** | 5.8% | **+0.21** |

Car 01's signature is an **imbalance between its two refrigeration circuits** —
one circuit running 243 kPa lower on the high side than its twin — together
with the consist's highest half-load duty (58.7%). That is what an undercharged
circuit looks like when the unit's second, healthy circuit partially
compensates: the cabin stays nearly on setpoint, so the thermal signal is
weak. Car 04, by contrast, is simply the hardest-working unit: the highest
pressure ratio, the lowest suction, 89% full cooling, and the warmest cabin —
high load, not undercharge.

**This is a genuine limitation, not a tuning opportunity.** The discriminating
evidence is per-circuit pressure and half-load duty, and neither exists in
`acv_test_case.xlsx`, whose running mode never distinguishes half from full
cooling. A feature added to rescue case_04 would be inert on the test file and
justified by a single journey. It was not added. Linear rank decay is also
forgiving here by design: this failure mode costs 0.125, not a zero.

---

## 6. Robustness of the test-case answer

All checks below are produced by `validate_physics_ranker.py`.

**B1 — weight perturbation.** The five weights are engineering judgement, so
they must not be load-bearing.

| Weighting | Labelled score | Test top-1 | Test ranking |
|---|---|---|---|
| default (1, .5, .25, .25, .25) | 0.9792 | **01** | `01\|03\|04\|08\|07\|06\|02\|05` |
| `mean_rel` alone | 0.9792 | **01** | `01\|04\|03\|07\|08\|06\|02\|05` |
| all five equal | 0.9792 | **01** | `01\|04\|03\|08\|07\|06\|02\|05` |
| setpoint-heavy | 0.9792 | **01** | `01\|03\|04\|08\|07\|06\|02\|05` |
| peer-relative only (no setpoint term) | 0.9792 | **01** | `01\|04\|03\|08\|07\|06\|02\|05` |
| persistence-heavy | 0.9792 | **01** | `01\|03\|04\|08\|07\|06\|02\|05` |

**B2 — leave one feature out.** Deleting any single feature changes neither the
labelled score (0.9792 in all five cases) nor the top-ranked test car (01).
No feature is carrying the answer alone.

**B3 — standardisation.** Robust median/MAD and classical mean/σ both give
0.9792 and both put Car 01 first; robust widens the rank-1/rank-2 gap from
3.27 to 4.66.

**B4 — per-day stability.** Scoring each calendar day independently, the test
case returns Car 01 on 3 of 4 days (Car 04 on the first). Labelled journeys
behave the same way: whole-journey aggregation is more reliable than any single
day (case_02 3/4, case_03 3/4, case_05 1/4 yet correct in aggregate). The
verdict is not driven by one unusual day, and day-level noise is the reason the
model scores the whole journey.

Across 11 method variants and 2 standardisation schemes, **Car 01 is first
every time.** Ranks 2 and 3 swap between Car 03 and Car 04 depending on
variant — their health indices differ by 0.013, so that pair is genuinely a
coin flip and is reported as such.

---

## 7. Test-case prediction

```
file_id             ranked_cars
acv_test_case.xlsx  01|03|04|08|07|06|02|05
```

| Rank | Car | Health index | mean ΔT_rel | mean ΔT_set | p95 ΔT_rel | Elevated | Persistence 15 min |
|---|---|---|---|---|---|---|---|
| **1** | **01** | **5.837** | **+0.105** | **+0.125** | **1.00** | **42.0%** | **36.7%** |
| 2 | 03 | 1.178 | +0.060 | −0.053 | 0.50 | 28.4% | 15.9% |
| 3 | 04 | 1.165 | +0.062 | −0.079 | 0.75 | 26.9% | 14.6% |
| 4 | 08 | 0.019 | +0.004 | −0.088 | 0.50 | 26.8% | 16.3% |
| 5 | 07 | −0.187 | +0.013 | −0.076 | 0.50 | 22.8% | 9.1% |
| 6 | 06 | −1.005 | −0.008 | −0.113 | 0.50 | 20.5% | 8.3% |
| 7 | 02 | −2.465 | −0.053 | −0.173 | 0.50 | 19.0% | 4.1% |
| 8 | 05 | −3.203 | −0.084 | −0.185 | 0.50 | 15.5% | 2.5% |

Car 01 is the only car on the consist with a **positive** mean setpoint error:
every other car sits below its cooling setpoint, and Car 01 sits above it. It
is also the only car whose 95th-percentile peer deficit reaches 1.00 °C, and it
is elevated above its peers 42% of the time versus 15–28% for the rest. The
separation from rank 2 is 4.66 index points against a 0.013 gap between ranks
2 and 3 — the leading diagnosis is clear-cut even though the runners-up are
not distinguishable.

The absolute deltas are small (≈0.1 °C) because this is an **early-stage
leak**: capacity is degraded but the unit still roughly holds the cabin. The
signal is the consistency, not the magnitude — 7,872 co-cooling samples over
four days, elevated 42% of the time.

---

## 8. Corrections to the previous version of this document

| Previous claim | Actual |
|---|---|
| LOOCV ground truth `01, 04, 03, 03, 03, 08` | `Train_Labels.csv` says `01, 02, 03, 01, 04, 06`. Four of six rows were wrong. |
| "Case 05 is the single miss (true Car 03 ranked 2nd)" | Case 05 is a **top-1 hit** (true car is 04). The miss is **case_04** (true car 01, ranked 2nd). |
| Case 04 has 4 cars | Its headers declare **8**; only 4 report telemetry. All 8 must still be ranked, or the unranked ones score 0. |
| Test Car 01: ΔT_rel +0.82 °C, persistence 84.3% | Reproducible values are **+0.105 °C** and **36.7%**. The old table was not reproducible from the code. |
| Test ranking `01\|04\|03\|08\|07\|06\|02\|05`, margin 1.338 | `01\|03\|04\|08\|07\|06\|02\|05`, margin 4.659. Cars 03/04 differ by 0.013, i.e. a tie. Top-1 unchanged. |
| Sampling is "1 s/10 s" | 30 s for six files; 10 s for `acv_case_04` only. |
| Pairwise logistic model adds nonlinear skill | Measured contribution is exactly zero (0.9583 with and without). Removed. |
| `Load Halved` is an informative parameter | It reads `Normal` for every car in every file (`_audit_stab.txt`). It carries no information. |

A further bug was found and fixed while writing this: mapping
`ACV Operating Mode` as a data-validity flag (it also emits the token
`Invalid`) discarded 21,845 of `acv_case_04`'s 21,849 rows, leaving 4 rows, all
features zero, and a tied index whose sorted-order output *coincidentally*
placed the true faulty car first — a spurious 1.0000. Only
`ACV Information Valid` is a validity flag. After the fix, case_04 honestly
scores 0.875.

---

## 9. Reproduction

```powershell
# Prediction (the deliverable). ~25 s for the test case.
cd ACV
..\venv\Scripts\python.exe predict.py --input ..\PS3\02_Datasets\ACV\Test --output acv_predictions.csv

# With the full per-car diagnostic table
..\venv\Scripts\python.exe predict.py --input ..\PS3\02_Datasets\ACV\Test\acv_test_case.xlsx --output acv_predictions.csv --verbose

# Labelled-journey check plus all robustness probes
..\venv\Scripts\python.exe validate_physics_ranker.py

# Single-file inspection
..\venv\Scripts\python.exe acv_physics_ranker.py ..\PS3\02_Datasets\ACV\Train\acv_case_05.xlsx
```

`predict.py` accepts either a single case file or a directory, and emits exactly
`file_id,ranked_cars`, matching `PS3/04_Example_Submission/acv_predictions.csv`.
Verified: for all 7 case files the output is a duplicate-free permutation of
that file's own two-digit header identifiers, and directory mode and
single-file mode agree byte for byte.

`acv_case_04.xlsx` takes ~5 minutes with openpyxl. Installing
`python-calamine` makes `pandas.read_excel` use a much faster engine
automatically; the ranker picks it up if present and the result is unchanged.

---

## 10. Honest summary of limitations

1. **Six labelled journeys.** 0.9792 is a sanity check, not a generalisation
   estimate. No confidence interval is meaningful at n=6.
2. **Only 8 SCADA parameters at inference.** Multi-circuit leaks that the
   healthy circuit compensates for are weakly observable from cabin
   temperature alone. This is exactly the case_04 failure, and it would recur
   on a similar held-out case.
3. **Requires ≥3 co-cooling peers.** A consist where most units are stopped or
   ventilating yields few usable timestamps. Files report their usable sample
   count (`n_cooling_samples`) so this is visible rather than silent.
4. **A consist-wide leak would be missed** by the peer-relative term. The
   absolute `ΔT_set` term is the guard against this, which is why it is kept at
   weight 0.5 despite `ΔT_rel` alone scoring the same on these six journeys.
5. **Ranks 2 and 3 on the test case are not meaningfully separated** (0.013
   index points). Only the rank-1 diagnosis should be treated as actionable.
