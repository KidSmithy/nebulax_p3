# ACV2 — Physics-Informed Refrigerant-Leak Localisation

Diagnose **which car of an 8-car train has the refrigerant leak** from ACV telemetry,
for NebulaX 2026 Problem Statement 3 (ACV subsystem).

The system is a **peer-consensus anomaly detector driven by the vapour-compression
energy balance**, not a trained classifier. That choice is forced by the data: six
labelled files with one faulty car each give roughly six training examples, far too
few to fit a decision boundary over twenty features. What the data *does* provide in
abundance is within-file structure — one faulty car and seven healthy siblings
recorded under identical ambient, solar and service conditions — so the consist
itself is used as the reference model.

| Result | Value |
|---|---|
| Rank-decay score on the 6 labelled cases | **1.0000** (true faulty car ranked 1st in all 6) |
| Leave-one-file-out cross validation | **1.0000** |
| Random-permutation baseline (8 cars) | 0.5625 |
| Block-bootstrap top-1 stability | 0.80 – 1.00 per case |
| Held-out test prediction | `acv_test_case.xlsx → 01\|04\|03\|08\|07\|06\|02\|05` |
| Verification suite | 21/21 passing |

Full numbers: [`reports/evaluation.md`](reports/evaluation.md),
[`reports/feature_diagnostics.txt`](reports/feature_diagnostics.txt),
[`reports/inspect_test_case.txt`](reports/inspect_test_case.txt).

How it was built, step by step, including the dead ends and the comparison against the
pre-existing `ACV/` pipeline: [`METHODOLOGY_STEP_BY_STEP.md`](METHODOLOGY_STEP_BY_STEP.md).

---

## 1. Why a leak is visible at all

A saloon is a lumped thermal capacitance driven by ambient conduction and internal
gains and cooled by its own pack:

```
C dT_in/dt = UA (T_out − T_in) + Q_gain − Q_evap(t),     Q_evap ≤ Q_max = ṁ (h_out − h_in)
```

The refrigerant mass flow `ṁ` is set by compressor displacement and **suction
density**, which collapses when the circuit loses charge. An undercharged pack
therefore has a smaller `Q_max`. Setting `dT_in/dt = 0`:

```
T_in* = T_out + (Q_gain − Q_max) / UA
```

Three consequences define the feature set:

1. **Capacity deficit.** While `Q_max` still exceeds the load the controller simply
   modulates and the leak is invisible. Once the load exceeds `Q_max` the car settles
   above set point.
2. **Load dependence.** The shortfall grows with `(T_out − T_set)`. A *miscalibrated
   sensor* gives a constant offset with zero slope; a *capacity loss* gives a slope.
   This is what separates the two, and it is why `load_sensitivity` exists.
3. **Progression.** A leak is monotonic — the deficit grows across the days of the
   record, which a differently-set thermostat or a heavier passenger load does not.

Because the eight cars share the weather and the timetable, the **peer-relative**
elevation of one car against the median of its siblings cancels `UA`, `Q_gain` and the
ambient, leaving the capacity term. That is the entire reason localisation is possible
from cabin temperature alone.

---

## 2. The two evidence branches

### Thermal branch — works on every file

Derived purely from cabin temperature, set point, outdoor temperature and running
mode, so it runs on both schemas in the dataset.

| Feature | Physics |
|---|---|
| `elev_mean`, `elev_p90` | cabin temperature above the **leave-one-out** sibling median during cooling |
| `elev_load_stratified` | the same elevation averaged with equal weight over thermal-load bins, so an unequal duty history cannot fake it |
| `setpoint_error` | `T_in − T_set` while cooling: control demand the pack cannot meet |
| `elev_persistence` | fraction of cooling time with a sustained 15-minute elevation |
| `load_sensitivity` | `d(elevation)/d(T_out − T_set)` on bin means — capacity loss vs sensor bias |
| `capacity_shortfall` | `(T_in − T_set)/(T_out − T_set)`: dimensionless undelivered fraction of the demanded cooling span |
| `pulldown_rate` | peak sustained pull-down rate (K/h) on cooling ramps — probes `Q_max` directly, independent of set point |
| `greybox_cool_rate` | `Q_max/C` identified by least squares from `dT_in/dt = a(T_out − T_in) + b·u + c` |
| `elev_trend`, `cusum_fraction` | progression: K/day trend, and a CUSUM change point giving the **leak onset time** |
| `full_demand_frac` | how often the controller escalates to full cooling |
| `integrity_loss` | rate of invalid-status flags and implausible readings the pack reports about itself |

### Refrigerant branch — activates on the rich schema

One case file (`acv_case_04.xlsx`, 483 columns) exposes high- and low-side pressures
for **two independent refrigeration circuits per car**. Here the leak is measured
rather than inferred, and the key move is to use the **sibling circuit inside the same
pack** as the reference — it shares the ambient, the cabin *and* the demand command
exactly, which removes the duty-cycle confound that makes raw between-car pressure
comparison misleading.

| Feature | Physics |
|---|---|
| `circuit_asym_lift` | relative mismatch in pressure lift `p_high − p_low` (∝ compression work per unit mass) between the pack's two circuits |
| `circuit_asym_high` | mismatch in condensing pressure: an undercharged condenser holds less refrigerant |
| `circuit_asym_ratio` | mismatch in compression ratio `p_high/p_low`, which rises as the suction side collapses faster than the discharge side |
| `suction_excursion` | `p50 − p05` of suction pressure: intermittent evaporator **starvation** toward the low-pressure cut-out |
| `lift_deficit` | shortfall of the weakest circuit's demand-matched lift against the sibling cars |
| `compressor_duty`, `compressor_cycling` | run fraction and start rate — low-pressure short cycling |

That branch is what makes `acv_case_04` solvable. Its thermal evidence points to car
04 (warmest cabin, 89% full-cooling demand, lowest suction pressure), but the true
faulty car is 01 — whose *circuit 2* runs ~12% below its own circuit 1 on head
pressure, dips to 280 on suction against its own 680 median, and short-cycles at
3.4 compressor starts/hour versus 0.8–1.2 for its siblings. (The workbook does not
state pressure units; every indicator here is a ratio or an intra-pack difference, so
the diagnosis does not depend on knowing them.) Thermal evidence alone ranks car 01
2nd — zeroing the refrigerant group drops the overall mean from 1.000 to 0.979 for
exactly that reason.

---

## 3. From features to a ranking

```
raw feature ──robust z vs the other cars──► oriented by physics ──weighted pool──► score
                    (median / MAD)              (sign is fixed, never fitted)
```

1. **Robust z-scores** use median and MAD. The MAD is unaffected by the single
   outlier it exists to expose; an ordinary standard deviation is inflated by it.
2. **Orientation** is fixed in `config.FEATURE_SPEC` from the direction the energy
   balance predicts. No sign is learned from the labels.
3. **Weights** are a physical prior, renormalised over whichever channels the file's
   schema actually supports — a thin-schema file is scored on thermal evidence rather
   than penalised for missing pressure channels.
4. An **IsolationForest** over the same feature space contributes a model-free second
   opinion at 15% weight, signed by the physics projection so a car that is
   anomalously *good* can never be promoted.
5. Cars with no usable telemetry (four of eight in the rich case) are appended after
   every diagnosable car in header order — never interleaved, never dropped, since the
   submission must rank every car in the file.

Every verdict is traceable: `contributions` records the signed contribution of each
named physical channel per car.

---

## 4. Guarding against six-file overfitting

* **Nothing is fitted except six group-level multipliers**, on a coarse grid, by
  coordinate ascent. No per-feature sign, no per-file quantity, no estimator.
* **LOOCV re-runs that calibration from scratch** for each held-out file, so the
  reported 1.0000 never sees its own tuning. In practice the physical prior already
  scores 1.0000, so calibration returns the prior unchanged — the shipped model *is*
  the physics.
* **Group ablation** shows no single branch is load-bearing: zeroing any one group
  keeps the score at ≥0.979.
* **Block bootstrap** re-ranks each case on random 70% subsets of its 6-hour blocks.
  Whole blocks are kept or dropped together because consecutive 30-second samples are
  strongly autocorrelated and an i.i.d. bootstrap would grossly overstate the
  effective sample size.

Honest caveats:

* The refrigerant branch is validated on **one** file. Its internal logic is sound
  vapour-compression physics and it is cross-checked against three independent
  indicators that agree, but "n = 1" is the truth.
* `full_demand_frac` scores 0.531 on its own — *below* the 0.5625 random baseline —
  and `elev_trend` only 0.667. They carry 4% and 6% of the weight respectively and
  removing the whole `control` group changes nothing (ablation delta 0.000).
* `pulldown_rate` is coarse: cabin temperature is quantised to 0.5 K at 30 s
  sampling, so the rate lands on a small set of discrete values.
* `greybox_cool_rate` only identifies on `acv_case_06` (R² > 0.02). Elsewhere the
  packs cool 85–99% of the time, so the cooling indicator has too little variation to
  separate `UA/C` from `Q_max/C`. It reports "not identified" and contributes nothing
  rather than contributing noise.

---

## 5. Test-case verdict, and where it is uncertain

```
acv_test_case.xlsx → 01|04|03|08|07|06|02|05
```

Car 01 wins by a wide margin (score 2.34 vs 0.56) and is top-1 in **40/40** bootstrap
draws and under **every** group ablation. But the absolute signal is much weaker than
in any training case: +0.11 K mean elevation, against 0.23–1.31 K for the six labelled
faulty cars.

There is a genuine disagreement between channels worth stating plainly:

* **Thermal, capacity and progression evidence → car 01.** Highest elevation, largest
  setpoint error, positive load sensitivity, largest CUSUM integral (peak 23.9,
  onset day 1.9).
* **Data-integrity evidence → car 04.** 39 invalid/dropout events versus 0–4 for
  every other car. In the three training files where this channel fired it identified
  the faulty car correctly all three times.

The fusion puts 01 first and 04 second. Under rank-decay scoring the expected value of
that ordering versus the reverse is nearly identical (1.000/0.875 either way), so the
defensible choice is to keep both in the top two and let the thermodynamics — the
primary evidence of lost cooling capacity — set the order.

---

## 6. Repository layout

```
ACV2/
├── predict.py                  # submission CLI:  --input / --output
├── requirements.txt
├── acv2/
│   ├── config.py               # schema aliases, mode vocabularies, feature registry, thresholds
│   ├── io_loader.py            # schema-agnostic loader → CasePanel, Parquet cache
│   ├── cleaning.py             # physical-plausibility masking, operating context, gap segmentation
│   ├── physics.py              # the model: robust stats, grey-box ID, CUSUM, circuit thermodynamics
│   ├── features.py             # thermal + refrigerant feature extraction
│   ├── detectors.py            # robust z, physics pooling, IsolationForest, fusion
│   ├── ranker.py               # end-to-end ranking + human-readable verdict
│   ├── evaluate.py             # rank-decay metric, LOOCV, ablation, block bootstrap
│   └── reporting.py            # Markdown table writer
├── scripts/
│   ├── explore_schema.py       # signal inventory per file: what is populated and does it vary per car
│   ├── diagnose_features.py    # per-case feature tables + single-channel discrimination
│   ├── evaluate_loocv.py       # the full evaluation → reports/evaluation.md
│   ├── inspect_case.py         # deep dive on one file: contributions, stability, sensitivity
│   └── train.py                # calibrate + export artifacts/acv2_model.joblib
├── tests/test_acv2.py          # 21 checks: schema, cleaning, physics signs, output contract
├── artifacts/                  # model weights, feature dump, training summary
├── reports/                    # evaluation report, feature diagnostics, test-case inspection
└── cache/                      # Parquet cache (safe to delete)
```

---

## 7. Running it

```powershell
# from the repository root, using the project venv

# 1. inference — the deliverable
.\venv\Scripts\python.exe ACV2\predict.py --input PS3\02_Datasets\ACV\Test --output ACV2\acv_predictions.csv

# with a full per-car diagnosis and a JSON sidecar
.\venv\Scripts\python.exe ACV2\predict.py -i PS3\02_Datasets\ACV\Test -o ACV2\acv_predictions.csv --report

# 2. calibrate and export the weight artefact
.\venv\Scripts\python.exe ACV2\scripts\train.py

# 3. evaluation: prior, LOOCV, ablation, single-channel scores, bootstrap
.\venv\Scripts\python.exe ACV2\scripts\evaluate_loocv.py

# 4. deep dive on one file
.\venv\Scripts\python.exe ACV2\scripts\inspect_case.py PS3\02_Datasets\ACV\Test\acv_test_case.xlsx

# 5. verification suite (no pytest needed; pytest also works if installed)
.\venv\Scripts\python.exe ACV2\tests\test_acv2.py
```

`predict.py` runs correctly with **no** artefact present — it falls back to the
physical prior, which is what the reported scores were obtained with.

The first read of `acv_case_04.xlsx` takes ~3 minutes (34 MB, 483 columns, openpyxl);
it is cached to Parquet afterwards and subsequent loads take under a second.

---

## 8. Notes on the dataset

* **Two schemas.** Five files carry 8 parameters per car at 30 s sampling; one carries
  59 at 10 s. Nothing in the code assumes a parameter list — headers are parsed and
  mapped onto canonical signals, and unknown parameters are retained under a slugged
  name. `rows_for_minutes()` converts every physical window using each file's own `dt`,
  so a "15-minute persistence" window is 30 rows in one file and 90 in another.
* **Outdoor temperature is a train-level quantity.** Two of the files instrument only
  the end cars, so the ambient used in the load proxy is the consist median — which is
  also physically correct and removes single-sensor bias.
* **Data gaps are real.** 9–41 gaps per file, up to 11 hours. Rolling means and
  derivatives never cross a gap: they are segmented on a discontinuity larger than 5×
  the median sampling interval.
* **Cleaning only removes the physically impossible.** Cabin temperatures outside
  5–45 °C (the 0.0 °C dropouts) and rows the pack itself flags `Invalid` are masked.
  Nothing is smoothed or imputed in a way that could create or hide a temperature
  difference between cars — those differences *are* the diagnosis. The removed
  material is not discarded either: its rate becomes the `integrity_loss` channel.
