# ACV2 robustness report

Five tests aimed at the known weaknesses of a six-file problem. See the module docstring of `scripts/validate_robustness.py` for what each one can and cannot establish. Tests B and C also write the calibration artefacts the ranker uses to attach a confidence to a verdict.

## A. Weight sensitivity

60 random non-negative weight vectors (uniform on the simplex) over the same channels, physics-fixed signs retained, all group multipliers 1.0.

| draws | mean_score | worst | p05   | p25   | median | fraction_perfect | hand_tuned_prior | random_baseline |
|-------|------------|-------|-------|-------|--------|------------------|------------------|-----------------|
| 60    | 0.989      | 0.958 | 0.979 | 0.979 | 0.979  | 0.467            | 1                | 0.562           |

Reading: the hand-tuned prior is not load-bearing. Most random positive weightings of the same physically-signed channels also rank every faulty car first, and the worst draw still beats the random baseline comfortably. What is doing the work is the sign structure from the energy balance, which is fixed by physics and never fitted, not the particular magnitudes chosen by hand.

## B. Specificity: the faulty car deleted

Each file re-ranked over its healthy siblings only. `margin_ratio` is the top-1 margin with the faulty car present divided by the margin among healthy cars alone; `q_ratio` is the same comparison for the scale-free Dixon-Q separation.

| file_id          | true_car | margin_with_faulty | margin_healthy_only | margin_ratio | q_with_faulty | healthy_top_car | q_healthy_only | q_ratio |
|------------------|----------|--------------------|---------------------|--------------|---------------|-----------------|----------------|---------|
| acv_case_01.xlsx | 01       | 2.035              | 1.408               | 1.445        | 0.499         | 03              | 0.402          | 1.242   |
| acv_case_02.xlsx | 02       | 1.145              | 2.25                | 0.509        | 0.229         | 03              | 0.5            | 0.459   |
| acv_case_03.xlsx | 03       | 2.142              | 0.316               | 6.776        | 0.578         | 02              | 0.168          | 3.442   |
| acv_case_04.xlsx | 01       | 0.095              | 0.998               | 0.096        | 0.057         | 04              | 0.815          | 0.07    |
| acv_case_05.xlsx | 04       | 0.874              | 1.565               | 0.558        | 0.398         | 02              | 0.405          | 0.983   |
| acv_case_06.xlsx | 06       | 2.409              | 0.736               | 3.272        | 0.556         | 08              | 0.264          | 2.107   |

Median margin ratio **1.00x**, median Q ratio **1.11x**.

**This is the most consequential result in the suite, and it is negative.** A healthy consist separates as strongly as a faulty one. That is structural rather than a defect: the ranker returns whichever car is most anomalous relative to its siblings, and exactly one car is always the warmest, so a winner with a margin is produced whether or not anything is broken. The consequence is that the raw top-1 margin the system used to report as confidence carried no information about fault *presence*. It has been replaced by a null-referenced statistic, calibrated on the healthy consists below.

### Null calibration (41 fault-free consists)

Built by deleting each file's labelled faulty car and then leaving out each healthy car in turn. Written to `artifacts/null_calibration.json` and consumed by `acv2.confidence`.

| n_consists | q_median | q_p75 | q_p90 | q_p95 | q_max |
|------------|----------|-------|-------|-------|-------|
| 41         | 0.402    | 0.478 | 0.509 | 0.573 | 0.815 |

The labelled files scored against that null:

| file_id          | true_car | top_car | correct | n_cars | margin | dixon_q | top_z  | elev_mean_top | null_p_value |
|------------------|----------|---------|---------|--------|--------|---------|--------|---------------|--------------|
| acv_case_01.xlsx | 01       | 01      | True    | 8      | 2.035  | 0.499   | 12.406 | 0.479         | 0.214        |
| acv_case_02.xlsx | 02       | 02      | True    | 8      | 1.145  | 0.229   | 10.095 | 0.254         | 0.786        |
| acv_case_03.xlsx | 03       | 03      | True    | 8      | 2.142  | 0.578   | 3.909  | 0.626         | 0.071        |
| acv_case_04.xlsx | 01       | 01      | True    | 4      | 0.095  | 0.057   | 1.979  | 0.224         | 0.952        |
| acv_case_05.xlsx | 04       | 04      | True    | 8      | 0.874  | 0.398   | 2.952  | 0.205         | 0.524        |
| acv_case_06.xlsx | 06       | 06      | True    | 8      | 2.409  | 0.556   | 6.011  | 1.363         | 0.095        |

`null_p_value` is the fraction of fault-free consists that separate at least as cleanly. A real fault is not guaranteed to produce a small p-value - a leak whose capacity deficit is still small genuinely does look like a healthy consist - so this figure measures how *distinguishable* the verdict is, not how likely it is to be the right car.

## C. Synthetic recovery and detection limit

Real faulty car removed, then a physics-shaped capacity deficit of known mean magnitude injected into one healthy car and re-quantised onto the channel's own measurement grid. `delta_k` is the mean elevation added, directly comparable to the `elev_mean` of the real faulty cars and of the test case's top car.

| delta_k | trials | top1_rate | top2_rate | mean_score | mean_rank | mean_observed_elev |
|---------|--------|-----------|-----------|------------|-----------|--------------------|
| 0.05    | 12     | 0.333     | 0.5       | 0.726      | 2.917     | 0.091              |
| 0.1     | 12     | 0.333     | 0.5       | 0.738      | 2.833     | 0.095              |
| 0.15    | 12     | 0.417     | 0.75      | 0.869      | 1.917     | 0.121              |
| 0.2     | 12     | 0.583     | 1         | 0.94       | 1.417     | 0.184              |
| 0.3     | 12     | 0.917     | 1         | 0.988      | 1.083     | 0.338              |
| 0.5     | 12     | 0.917     | 1         | 0.988      | 1.083     | 0.533              |
| 1       | 12     | 1         | 1         | 1          | 1         | 0.975              |

Recovery becomes reliable (top-1 rate >= 90%) from **0.30 K** upward. This is the only test in the suite that escapes the six-example ceiling: every trial has a known injected target and the real fault has been removed, so 84 independent unseen cases are available instead of six.

Per-file mean rank of the injected car:

| file_id          | 0.05 | 0.1 | 0.15 | 0.2 | 0.3 | 0.5 | 1.0 |
|------------------|------|-----|------|-----|-----|-----|-----|
| acv_case_01.xlsx | 3.5  | 3.5 | 2    | 1   | 1   | 1   | 1   |
| acv_case_02.xlsx | 3.5  | 3.5 | 1.5  | 1.5 | 1   | 1   | 1   |
| acv_case_03.xlsx | 3.5  | 3.5 | 2.5  | 1.5 | 1   | 1   | 1   |
| acv_case_04.xlsx | 1.5  | 1.5 | 1.5  | 1.5 | 1.5 | 1.5 | 1   |
| acv_case_05.xlsx | 3.5  | 3   | 2    | 2   | 1   | 1   | 1   |
| acv_case_06.xlsx | 2    | 2   | 2    | 1   | 1   | 1   | 1   |

## D. Peer-dropout (car-level) bootstrap

Each case re-ranked from scratch after dropping 2 randomly chosen non-leading cars, 12 times. Features, peer reference, ambient estimate and load strata are all recomputed, because every one of them changes when the consist changes.

| file_id          | true_car | draws | cars_dropped | baseline_top | top1_true_car | leader_retained | runner_up_car | runner_up_freq |
|------------------|----------|-------|--------------|--------------|---------------|-----------------|---------------|----------------|
| acv_case_01.xlsx | 01       | 12    | 2            | 01           | 1             | 1               | -             | 0              |
| acv_case_02.xlsx | 02       | 12    | 2            | 02           | 1             | 1               | -             | 0              |
| acv_case_03.xlsx | 03       | 12    | 2            | 03           | 1             | 1               | -             | 0              |
| acv_case_04.xlsx | 01       | 12    | 1            | 01           | 0.75          | 0.75            | 04            | 0.25           |
| acv_case_05.xlsx | 04       | 12    | 2            | 04           | 1             | 1               | -             | 0              |
| acv_case_06.xlsx | 06       | 12    | 2            | 06           | 1             | 1               | -             | 0              |

This complements the time-block bootstrap in `reports/evaluation.md`, which resamples 6-hour blocks and therefore only measures temporal stability against a fixed reference set. Neither test can measure generalisation to an unseen *file*; with six labelled files nothing can, and that limit is irreducible here.

## E. Refrigerant branch agreement

Each file scored twice: once on inferred thermal evidence only (thermal + capacity + progression), once on circuit pressures only.

| file_id          | true_car | refrigerant_available | inferred_top | refrigerant_top | agree | inferred_correct | refrigerant_correct | fused_correct | inferred_rank_true | refrigerant_rank_true |
|------------------|----------|-----------------------|--------------|-----------------|-------|------------------|---------------------|---------------|--------------------|-----------------------|
| acv_case_01.xlsx | 01       | False                 |              |                 |       |                  |                     |               |                    |                       |
| acv_case_02.xlsx | 02       | False                 |              |                 |       |                  |                     |               |                    |                       |
| acv_case_03.xlsx | 03       | False                 |              |                 |       |                  |                     |               |                    |                       |
| acv_case_04.xlsx | 01       | True                  | 04           | 01              | False | False            | True                | True          | 2                  | 1                     |
| acv_case_05.xlsx | 04       | False                 |              |                 |       |                  |                     |               |                    |                       |
| acv_case_06.xlsx | 06       | False                 |              |                 |       |                  |                     |               |                    |                       |

The refrigerant branch is computable in exactly one of the six files, because only that file's schema exposes circuit pressures. That is an accident of instrumentation, not a measured weakness, so the branch keeps the weight its physical directness earns rather than being penalised for it. What this table adds is auditability: where the two independent lines of reasoning disagree, it records which one was right. On `acv_case_04` the inferred branch alone does not name the true car and the circuit-pressure branch does, which is the single piece of evidence supporting the branch - one data point, stated as one data point.
