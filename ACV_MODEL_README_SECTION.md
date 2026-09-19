# ❄️ ACV Subsystem — Refrigerant-Leak Localisation

> Which car on an 8-car consist is losing refrigerant charge?
> Physics-informed peer-consensus ranking. **No trained model, no pickle, zero fitted
> parameters.**

---

## What it does

Given one consist telemetry file — CSV or Excel — the model ranks **every car in that
file's headers** from most to least likely to be leaking refrigerant.

```
acv_test_case.xlsx  →  01|04|03|08|07|06|02|05
```

Reference implementation: `ACV2/acv2/`. Backend adapter: `backend/models/acv_model.py`.

---

## Why a leak is visible at all

A car's saloon is a lumped thermal capacitance driven by ambient conduction, solar and
passenger gains, and cooled by its own ACV pack:

$$C \frac{dT_{in}}{dt} = UA\,(T_{out} - T_{in}) + Q_{gain} - Q_{evap}(t)$$

Evaporator duty is bounded by the pack's refrigeration capacity,
$Q_{evap} \le Q_{max} = \dot{m}\,(h_{out} - h_{in})$, and the refrigerant mass flow
$\dot{m}$ is set by compressor displacement and **suction density**, which collapses when
the circuit loses charge. So an undercharged pack has a smaller $Q_{max}$. At steady state:

$$T_{in}^{*} = T_{out} + \frac{Q_{gain} - Q_{max}}{UA}$$

Three consequences drive every feature:

1. **Capacity deficit** — once the load exceeds $Q_{max}$, the cabin settles above set point.
2. **Load dependence** — the shortfall grows with $(T_{out} - T_{set})$. A *miscalibrated
   sensor* gives a constant offset with zero slope; a *capacity loss* gives a slope. This is
   what separates the two.
3. **Progression** — a leak is monotonic; a differently-set thermostat is not.

**The key move:** the eight cars of one consist share ambient, solar and service pattern at
every instant, so the peer consist is a free reference model. Subtracting the
**leave-one-out median** of the sibling cars cancels $UA$, $Q_{gain}$ and the weather,
leaving the capacity term. That is the whole reason localisation is possible from cabin
temperature alone.

---

## Evidence channels (20 across 6 groups)

| Group | Channels | What it measures |
|---|---|---|
| **thermal** | `elev_mean`, `elev_load_stratified`, `elev_p90`, `elev_persistence`, `setpoint_error` | the capacity deficit itself, peer-relative and load-stratified |
| **capacity** | `load_sensitivity`, `capacity_shortfall`, `pulldown_rate`, `greybox_cool_rate` | leak vs sensor bias; grey-box identification of $Q_{max}/C$ |
| **progression** | `elev_trend`, `cusum_fraction` | K/day drift, and a CUSUM change point giving **leak onset time** |
| **refrigerant** | `circuit_asym_lift`, `circuit_asym_high`, `circuit_asym_ratio`, `suction_excursion`, `lift_deficit`, `compressor_duty`, `compressor_cycling` | direct circuit pressures (rich schema only) |
| **integrity** | `integrity_loss` | the pack's own admission that something is wrong |
| **control** | `full_demand_frac` | **descriptive only, zero weight** (see below) |

Each channel is converted to a **robust z-score (median/MAD)** against the other cars in the
same file, oriented by physics — the sign of every feature is fixed by the energy balance
and is *never* fitted — then pooled with a fixed physical prior, renormalised over whichever
channels the file's schema supports. A thin-schema file is scored on thermal evidence alone
rather than penalised for missing pressure sensors.

MAD rather than standard deviation because **a standard deviation is contaminated by the
very outlier being sought**: one leaking car inflates σ and drags the mean toward itself,
shrinking its own z-score.

### Schema agnosticism

The dataset ships two column layouts — 8 parameters per car in five files, 59 in one. The
loader reads each file's own headers and maps them onto canonical signal names, so both
load on their own terms. Only the 483-column file exposes high- and low-side pressures for
**two independent refrigeration circuits per car**; there the sibling circuit *inside the
same pack* is an even better reference than a sibling car, because it shares ambient, cabin
and demand command exactly.

---

## Results

Competition metric is a linear rank-decay score, $(n - (r-1))/n$. A random permutation of 8
cars scores **0.5625**, which every result below is reported against.

| | score |
|---|---|
| Physical prior, nothing fitted | **1.0000** |
| Leave-one-file-out cross validation | **1.0000** |
| Random-permutation baseline | 0.5625 |

All six labelled cases rank the true faulty car **first**. LOOCV re-selects the group
weights from scratch on the other five files for each held-out file, so the held-out score
never sees its own tuning. Because the prior already scores 1.0000, calibration returns it
unchanged — **the shipped model is the physics, not a fit to six files.**

### Robustness

| Test | Result |
|---|---|
| Group ablation | zeroing any one group keeps the score ≥ 0.979 |
| Weight sensitivity (400 random simplex draws) | worst 0.833, median 0.979 — the hand-chosen magnitudes are **not load-bearing**; the physics-fixed signs are |
| Time-block bootstrap (6-hour blocks, 70% subsets) | true car top-1 in 0.78–1.00 of draws |
| Peer-dropout bootstrap (consist resampled) | leader retained in 0.77–1.00 of draws |
| Synthetic recovery | 126 unseen injected cases — the only test that escapes the six-example ceiling |

---

## Confidence is calibrated, not read off the margin

This is the part most worth knowing, because the obvious approach is wrong and we measured
it.

The natural confidence signal is the top-1 margin, $s_1 - s_2$. **It carries no information
about whether a fault is present.** Deleting the labelled faulty car from each file and
re-ranking the healthy siblings produces a margin of *the same size* — median ratio
**1.00×**, and on one case the healthy-only consist separates ten times more strongly than
the genuine fault.

The cause is structural, not a bug: a peer-consensus ranker returns whichever car is most
anomalous relative to its siblings, and **exactly one car is always the warmest**, so a
winner with a margin appears whether or not anything is broken. A margin measures *spread*,
not *fault presence*.

Confidence is therefore referenced to consists **known to contain no fault**:

- **41 fault-free consists**, built by deleting each labelled faulty car and then leaving out
  each healthy car in turn.
- Two scale-free separation statistics, because measurement shows they are complementary:
  **Dixon's Q** $(s_1-s_2)/(s_1-s_n)$, a ratio of gaps, and **top-z**, the leader's distance
  in robust sigmas. Q is blind to how far out the leader actually is and saturates once
  z-clipping engages; top-z says nothing about whether the runner-up is nearly as bad.
- Both are tested and the smaller p-value taken, with a **Bonferroni factor of two** to pay
  honestly for looking twice.

| file | true car | Q | top-z | p via Q | p via top-z | combined p | level |
|---|---|---|---|---|---|---|---|
| acv_case_01 | 01 | 0.499 | 12.41 | 0.214 | 0.048 | **0.095** | moderate |
| acv_case_02 | 02 | 0.229 | 10.10 | 0.786 | 0.071 | **0.143** | moderate |
| acv_case_03 | 03 | 0.578 | 3.91 | 0.071 | 0.476 | **0.143** | moderate |
| acv_case_04 | 01 | 0.057 | 1.98 | 0.952 | 0.762 | **1.000** | weak |
| acv_case_05 | 04 | 0.398 | 2.95 | 0.524 | 0.619 | **1.000** | weak |
| acv_case_06 | 06 | 0.556 | 6.01 | 0.095 | 0.333 | **0.190** | moderate |

Q alone catches cases 03 and 06; top-z alone catches 01 and 02. Together, **4 of 6**.

**Two of six genuine faults still do not separate distinguishably from a healthy consist.**
That is the honest state of the evidence, not a defect — a leak whose capacity deficit is
still small genuinely does look like a healthy pack — and the point of reporting it is so the
system can say so instead of quoting a confident number it cannot justify.

### Detection limit

Recovery of a known injected deficit, measured over 126 trials:

| injected deficit | 0.05 K | 0.10 K | 0.15 K | 0.20 K | 0.30 K | 0.50 K | 1.00 K |
|---|---|---|---|---|---|---|---|
| top-1 recovery | 0.22 | 0.22 | 0.28 | 0.56 | 0.83 | 0.83 | 1.00 |

A verdict whose leading evidence falls below the magnitude at which recovery becomes
reliable is reported as provisional however cleanly it separates.

### One channel was demoted to zero weight

`full_demand_frac` measured **0.531** used alone — *below* the 0.5625 random baseline — is
computable in only 4 of 6 files, and its mean z at the true faulty car is **negative**
(−1.241): the faulty car is commanded to full cooling *less* often than its siblings. The
demand tier is a controller state driven by set point and load schedule, not by delivered
capacity, so it tracks duty history rather than health. It is still computed and reported as
descriptive, but it carries **zero weight** and is excluded from the unsupervised member too,
so "zero weight" means what it says.

---

## Backend integration

`backend/models/acv_model.py` is a thin adapter over `ACV2/acv2/`. The physics lives in
exactly one place.

### Input → output

Raw `bytes` in (exactly what `/api/predict/upload` passes), format chosen from the filename
extension so a binary `.xlsx` workbook is not forced through a CSV parser:

```
EXCEL (.xlsx)  2.22 MB  →  ['01','04','03','08','07','06','02','05']
CSV   (.csv)   4.85 MB  →  ['01','04','03','08','07','06','02','05']
```

The ranking is always a **full permutation** of every car in the file's own headers, using
the two-digit IDs exactly as the headers spell them. Cars with no usable telemetry are not
dropped — they are appended to the tail and flagged `assessable: false`, which matters on the
rich-schema file where only 4 of 8 cars are instrumented.

### Response shape

| method | purpose | `ranked_cars` |
|---|---|---|
| `predict_from_csv(bytes, name)` | single upload — the API surface | `list` |
| `predict_batch([(name, bytes)], zip)` | ZIP of trip logs | `list` |
| `localise_leak(path \| DataFrame)` | full diagnosis | `str` (`'01\|04\|…'`), plus `ranked_cars_list` |
| `evaluate_telemetry(...)` | live 10 Hz twin path, single pack | n/a |

Alongside the ranking, every response carries `confidence_level`
(`strong`/`moderate`/`weak`/`uncalibrated`), `separation_q`, `null_p_value`,
`detection_limit`, a `competing_hypothesis` block naming the runner-up and the channels that
favour it, and `car_diagnostics` — one row per car with named physical evidence and
plain-language labels for the HUD.

### Verdict banding

| calibrated level | status | action |
|---|---|---|
| strong / moderate | `ACTION_NEEDED` | `ACTION_REPLACE_FILTER` |
| weak | `WATCH` | `ACTION_INSPECT_ACV_PACK` — suspect still named, uncertainty stated |
| no car assessable | `GOOD` | `NONE` |

A weakly separated ranking is **not** promoted to a callout, and **not** demoted to "all
normal" either: it is still the best available ordering of the evidence.

### No model to load

```
model_artefact              : None
fitted_parameters           : 0
requires_training_data      : False
```

ACV is the only subsystem in this project that loads nothing at startup — Door, SHM and Rail
each unpickle a fitted bundle. Two optional JSON artefacts
(`null_calibration.json`, `detection_limit.json`) supply the confidence calibration; without
them the ranking is unchanged and confidence degrades to `uncalibrated` rather than a number
being invented.

---

## Verification

| suite | result |
|---|---|
| `ACV2/tests/` | **37 passed** |
| `backend/models/test_acv_leak_model.py` | **19 passed** |
| Backend integration, real files through `InferenceBroker().acv_model` | **5/5** labelled cases localised correctly |

The backend suite asserts the adapter reproduces ACV2's ranking, separation and p-value
exactly, so the HUD can never disagree with the reference implementation, and it guards
against a local copy of the physics being reintroduced.

```powershell
# rank a consist
python ACV2/predict.py -i PS3\02_Datasets\ACV\Test -o ACV2\acv_predictions.csv --report

# reproduce every number above
python ACV2\scripts\train.py                    # prior + LOOCV
python ACV2\scripts\validate_robustness.py      # 5 robustness tests + calibration artefacts
python ACV2\scripts\evaluate_loocv.py           # ablation, per-channel, bootstraps
```

---

## Honest limitations

- **Six labelled files remain six labelled files.** Synthetic recovery adds 126 unseen cases,
  but they derive from the same six records and the injection has the shape the features are
  built to see. It measures sensitivity to a deficit of a given size in real telemetry noise,
  not whether the leak model itself is right.
- **The channel set was chosen with all six files visible.** Weight sensitivity shows the
  magnitudes do not matter; it cannot show the feature engineering would have looked the same
  on unseen data.
- **No file-level bootstrap is possible.** Cross-case reliability is unmeasurable at n = 6.
- **The refrigerant branch is computable in one file**, because only that file's schema has
  pressure sensors. Rather than downweight the most physically direct evidence available for
  an accident of instrumentation, the branch is audited: on `acv_case_04` the inferred
  thermal branch names the wrong car and the circuit-pressure branch names the right one.
  That is one data point, reported as one data point.
- **The test-case verdict is weakly separated.** Car 01 is top-1 under every ablation and in
  every bootstrap draw, so the *ordering* is robust — but its +0.11 K deficit sits in the 22%
  recovery band. The ordering and the strength of the evidence are separate claims, and the
  system reports both.
