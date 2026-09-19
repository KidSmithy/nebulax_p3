# ACV2 — Audit response and change log

Response to a five-point critique of the refrigerant-leak localiser. Each criticism was
turned into a measurement rather than a paragraph of reassurance. Two of those
measurements came back negative, and one of them exposed a problem that was not on the
original list and was worse than anything on it.

**The submission string did not change. What changed is that the system now states how
much it is worth.**

```
acv_test_case.xlsx → 01|04|03|08|07|06|02|05
Confidence: WEAK   Q=0.486   p_vs_healthy=0.262   (null n=41)
```

Verified: 36/36 tests pass, physical prior still scores 1.0000, LOOCV still 1.0000.

---

## At a glance

| # | Criticism | Verdict after measurement | Change made |
|---|---|---|---|
| — | *(not on the list)* margin reported as confidence | **Confirmed, and serious** | Margin replaced by null-referenced Dixon-Q; new `acv2/confidence.py` |
| 1 | Prior hand-designed on all six files | Confirmed, but weights are not load-bearing | 400-draw weight-sensitivity test |
| 2 | Refrigerant branch validated on n=1 | Confirmed; reframed as instrumentation limit | Branch-agreement audit, no arbitrary downweight |
| 3 | `full_demand_frac` below random, still weighted | **Confirmed; worse than stated** | Demoted to zero weight + 2 follow-on fixes |
| 4 | Car 01 vs car 04 ambiguity | Confirmed and now quantified | Competing-hypothesis block + detection limit |
| 5 | Bootstrap is time-block only | Confirmed | New car-level peer-dropout bootstrap |

---

## 0. The finding that was not on the list

The system reported the raw top-1 margin as though it were confidence. It carries no
such information.

Test B deletes each file's labelled faulty car and re-ranks the healthy siblings that
remain. If the margin measured fault presence, it should collapse. It does not:

| file | margin with faulty car | margin healthy-only | ratio | Q with faulty | Q healthy-only |
|---|---|---|---|---|---|
| acv_case_01 | 2.035 | 1.408 | 1.45 | 0.499 | 0.402 |
| acv_case_02 | 1.145 | 2.250 | 0.51 | 0.229 | 0.500 |
| acv_case_03 | 2.142 | 0.316 | 6.78 | 0.578 | 0.168 |
| **acv_case_04** | **0.095** | **0.998** | **0.10** | **0.057** | **0.815** |
| acv_case_05 | 0.874 | 1.565 | 0.56 | 0.398 | 0.405 |
| acv_case_06 | 2.409 | 0.736 | 3.27 | 0.556 | 0.264 |

**Median margin ratio 1.00×.** On `acv_case_04` a consist of healthy cars separates ten
times more strongly than the genuine fault does.

The cause is structural, not a bug. A peer-consensus detector returns whichever car is
most anomalous relative to its siblings, and exactly one car is always the warmest. A
winner with a margin is produced whether or not anything is broken. **The margin measures
spread, not fault presence**, so "car 01 wins by a wide margin, 2.34 vs 0.56" was never
evidence that a fault exists.

### Fix — `acv2/confidence.py` (new)

1. **Dixon's Q**, `(s1 − s2) / (s1 − sn)`, replaces the margin. Scale-free, so unlike the
   margin it is comparable between a five-channel thin-schema file and a twelve-channel
   rich-schema one, and it is the classical small-sample test for a single outlier —
   exactly this problem's structure.
2. **A fault-free null**: 41 consists known to contain no fault, built by deleting each
   labelled faulty car and then leaving out each healthy car in turn. Confidence is
   `P(Q_null ≥ Q_observed)`.
3. **A detection-limit cross-check** (§4) that can downgrade a cleanly-separated verdict
   whose underlying deficit is too small to trust.

Null distribution (41 consists): median Q 0.402, p75 0.478, p90 0.509, p95 0.573, max 0.815.

The six labelled files against that null:

| file | true car | Q | p vs healthy | `elev_mean` |
|---|---|---|---|---|
| acv_case_01 | 01 | 0.499 | 0.21 | +0.48 K |
| acv_case_02 | 02 | 0.229 | 0.79 | +0.25 K |
| acv_case_03 | 03 | 0.578 | 0.07 | +0.63 K |
| acv_case_04 | 01 | 0.057 | 0.95 | +0.22 K |
| acv_case_05 | 04 | 0.398 | 0.52 | +0.21 K |
| acv_case_06 | 06 | 0.556 | 0.10 | +1.36 K |

**Three of six genuine faults are not distinguishable from a healthy consist.** That is
the honest state of the evidence, not a defect — a leak whose capacity deficit is still
small genuinely does look like a healthy pack. Reporting it is the point.

Artefacts (`artifacts/null_calibration.json`, `artifacts/detection_limit.json`) are built
by `scripts/validate_robustness.py`. If absent, the ranker reports `uncalibrated` and
attaches no probability rather than inventing one — verified by removing them and
re-running.

---

## 1. "The prior was hand-designed after inspecting all six files"

Correct, and LOOCV does not repair it: LOOCV re-runs the *calibration* on five files but
not the manual feature engineering or the weight choices.

So the weights were probed directly — 400 random non-negative weight vectors drawn uniform
on the simplex over the same physically-signed channels, all group multipliers at 1.0:

| draws | mean | worst | p05 | median | fraction scoring 1.0000 | baseline |
|---|---|---|---|---|---|---|
| 400 | 0.988 | 0.833 | 0.979 | 0.979 | 0.477 | 0.5625 |

The hand-tuned magnitudes are **not load-bearing**. What does the work is the sign
structure from the vapour-compression energy balance, which is fixed by physics and never
fitted. The objection loses most of its force but does not vanish: the *choice of
channels* was still made with all six files visible.

---

## 2. "The refrigerant branch is validated on exactly one file"

Correct, but the framing matters. The branch is computable in one file because only that
file's schema exposes circuit pressures. That is an accident of instrumentation, not a
measured weakness, and downweighting the most physically direct evidence available because
few of our six files carried pressure sensors would be the wrong response — weight should
track physical directness.

**No weight change was made.** The branch was made auditable instead. Test E scores each
file twice, on inferred thermal evidence alone and on circuit pressures alone:

| file | inferred branch top | circuit branch top | truth | which was right |
|---|---|---|---|---|
| acv_case_04 | 04 | 01 | 01 | circuit pressures |

The circuit branch is what gets this file right; the inferred branch alone does not. That
is the entire evidence for the branch — one data point, reported as one data point rather
than hidden in a weight.

---

## 3. "`full_demand_frac` is below random and still gets weight"

Correct, and the evidence against it was worse than the single number given:

- 0.531 used alone against a 0.5625 random baseline;
- computable in only 4 of 6 files;
- mean z at the true faulty car is **negative** (−1.241) — the faulty car is commanded to
  full cooling *less* often than its siblings.

Likely reason: the demand tier is a controller state driven by set point and load
schedule, not by delivered capacity, so it tracks duty history rather than health.

**Demoted 0.30 → 0.00.** Still computed and printed as descriptive; it cannot reach a
ranking. Two follow-on fixes were required to make that true rather than nominal:

- **`detectors.fuse` now restricts the IsolationForest to weighted channels only.**
  Previously a zero-weight channel still entered the unsupervised member and could move
  the ranking through the back door.
- **`ranker.load_model` now warns and overrides on a stale artefact.** This was a live
  bug: artefact `feature_weights` are merged *over* the config defaults, so the demotion
  silently did nothing until training was re-run.
- `features.extract` no longer lists a zero-weight group in `active_groups`, so the report
  cannot advertise `control` as evidence. It is surfaced as `descriptive_only_groups`.

Verified unchanged afterwards: prior 1.0000, LOOCV 1.0000, test-case prediction identical.

---

## 4. "Car 01 vs car 04 — the biggest practical uncertainty"

Now quantified rather than argued, via the synthetic-recovery test. The real faulty car is
removed, a physics-shaped capacity deficit of known mean magnitude is injected into a
healthy car and re-quantised onto the channel's own measurement grid, then recovery is
measured. This is the only test in the suite that escapes the six-example ceiling —
**126 independent unseen cases instead of six**.

| injected `delta_k` | top-1 rate | top-2 rate |
|---|---|---|
| 0.05 K | 0.22 | 0.33 |
| 0.10 K | 0.22 | 0.33 |
| 0.15 K | 0.28 | 0.61 |
| 0.20 K | 0.56 | 0.83 |
| 0.30 K | 0.83 | 0.89 |
| 0.50 K | 0.83 | 1.00 |
| 1.00 K | 1.00 | 1.00 |

Car 01's `elev_mean` is **0.1135 K**, which sits in the **22%** band. Reliable recovery
(≥90%) only begins at **1.00 K**. The six labelled faulty cars sit at 0.21–1.36 K.

So two separate claims now get reported separately:

- **The ordering is robust.** Car 01 is top-1 under every group ablation, in 40/40
  time-block bootstrap draws and 30/30 peer-dropout draws.
- **The evidence is thin.** Q = 0.486 is matched or exceeded by 26% of known-healthy
  consists, and the deficit is far below the measured detection limit.

`ranker._contest` emits an explicit competing hypothesis in `predict.py --report` and in
the JSON sidecar: every weighted channel is reported on the side it favours, with the
share of weighted evidence pointing the other way.

```
Competing hypothesis: car 04. The evidence is split — capacity, progression, thermal
favour car 01, while integrity, progression favour car 04 (17% of the weighted evidence).
  favours car 04: integrity_loss    z +4.00 vs -0.74
  favours car 04: cusum_fraction    z +2.77 vs +1.67
```

`integrity_loss` is deliberately secondary — it is not a measurement of cooling capacity
and it fires in only half the labelled files — but it is exactly what an electrical or
sensor fault would look like. **If the true defect is electrical rather than a loss of
charge, car 04 is the better answer.** The fusion keeps 01 first because the
thermodynamics are the primary evidence of lost cooling capacity, and under rank-decay
scoring the expected value of that ordering versus the reverse is nearly identical
(1.000/0.875 either way).

The output states the distinguishing test: inspect car 01 for charge loss (sight glass,
subcooling, weighed charge) and car 04 for wiring and sensor integrity.

---

## 5. "Bootstrap is time-block only, not file-level"

Correct. Resampling 6-hour blocks handles autocorrelation, but every draw still judges each
car against the same seven siblings, so it measures temporal stability and nothing else.

Added `evaluate.peer_dropout_stability`: drop non-leading cars from the consist and re-rank
from scratch, with features, peer reference, ambient estimate and load strata all
recomputed — every one of them changes when the consist changes.

| file | time-block bootstrap | peer-dropout bootstrap |
|---|---|---|
| acv_case_01 | 1.00 | 1.00 |
| acv_case_02 | 1.00 | 1.00 |
| acv_case_03 | 1.00 | 1.00 |
| acv_case_04 | 0.78 | 0.77 |
| acv_case_05 | 0.90 | 1.00 |
| acv_case_06 | 1.00 | 1.00 |

`acv_case_04` is lower on both because it has only four instrumented cars, so one dropout
is a quarter of its reference set.

**Neither test measures generalisation to an unseen *file*.** With six labelled files
nothing can. That limit is irreducible and is stated rather than worked around.

---

## Files changed

### New

| file | purpose |
|---|---|
| `acv2/confidence.py` | Dixon-Q separation, fault-free null p-value, detection-limit check, verdict banding |
| `artifacts/null_calibration.json` | 41 fault-free consists — the reference distribution for confidence |
| `artifacts/detection_limit.json` | measured recovery-versus-magnitude curve |
| `reports/robustness.md` | output of all five audit tests |

### Modified

| file | change |
|---|---|
| `acv2/config.py` | `full_demand_frac` → 0.00 with the measurement recorded; `CONFIDENCE` block; artefact paths |
| `acv2/detectors.py` | `fuse` restricts IsolationForest to weighted channels |
| `acv2/evaluate.py` | `peer_dropout_stability`, `branch_agreement`, `null_consists`, `observed_separation` |
| `acv2/features.py` | `active_groups` excludes zero-weight channels; adds `descriptive_only_groups` |
| `acv2/ranker.py` | confidence wiring, `_contest` competing hypothesis, `_stale_weights` guard, verdict no longer quotes the margin |
| `predict.py` | prints confidence and competing hypothesis; both in the JSON sidecar |
| `scripts/validate_robustness.py` | five tests; writes both calibration artefacts |
| `scripts/evaluate_loocv.py` | new §7 peer-dropout, §8 calibrated confidence |
| `scripts/train.py` | records zero-weight channels and confidence provenance; prints separation table |
| `tests/test_acv2.py` | 21 → **36** checks |
| `README.md` | new §4a on calibrated confidence; §4 and §5 rewritten |
| `METHODOLOGY_STEP_BY_STEP.md` | new §19, the audit, in seven subsections |

### New tests (15)

Zero-weight demotion is real (including that the IsolationForest cannot circumvent it, and
that a stale artefact cannot override it); Dixon-Q is scale-free and undefined below n=3;
p-value bounded and never exactly zero; confidence degrades to `uncalibrated` without
artefacts; confidence downgrades below the detection limit; verdict text no longer contains
the word "margin"; competing-hypothesis signs are consistent; split evidence on the test
case is reported; `without_cars` genuinely shrinks the consist; peer-dropout returns a valid
distribution; the refrigerant branch is what gets `acv_case_04` right.

---

## Reproduce

```powershell
& .\venv\Scripts\Activate.ps1

python -m pytest ACV2/tests -q                       # 36 passed
python ACV2/scripts/train.py                         # prior 1.0000, LOOCV 1.0000
python ACV2/scripts/validate_robustness.py --draws 400 --trials 3 --peer-draws 30
python ACV2/scripts/evaluate_loocv.py --draws 40
python ACV2/predict.py -i PS3\02_Datasets\ACV\Test -o ACV2\acv_predictions.csv --report
```

Run `validate_robustness.py` before `evaluate_loocv.py`: the former writes the calibration
artefacts the latter reports against.

---

## What the audit could not fix

- **Six labelled files remain six labelled files.** The synthetic-recovery test adds 126
  unseen cases but they are derived from the same six records, and the injection has the
  shape the features are designed to see. It measures sensitivity to a deficit of a given
  size in real telemetry noise, not whether the leak model itself is right.
- **The channel set was chosen with all six files visible.** Weight sensitivity shows the
  magnitudes do not matter; it cannot show that the feature engineering would have looked
  the same on unseen data.
- **No file-level bootstrap is possible.** Cross-case reliability is unmeasured and
  unmeasurable at n=6.
- **The refrigerant branch still rests on one file.** It is now audited and its one
  supporting data point is on the record, but n=1 is n=1.
- **The test case's evidence is genuinely weak.** The verdict is the best available
  ordering of thin evidence. Making the system say so was the fix; making the evidence
  stronger is not possible from this file.
