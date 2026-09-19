# ACV2 evaluation report

## 1. Uncalibrated physical prior

Weights are the physical prior in `config.FEATURE_SPEC`; nothing is fitted.

| file_id          | true_car | rank | n_ranked | score | ranked_cars                    |
|------------------|----------|------|----------|-------|--------------------------------|
| acv_case_01.xlsx | 01       | 1    | 8        | 1     | 01\|03\|04\|02\|07\|08\|06\|05 |
| acv_case_02.xlsx | 02       | 1    | 8        | 1     | 02\|03\|08\|07\|06\|04\|01\|05 |
| acv_case_03.xlsx | 03       | 1    | 8        | 1     | 03\|02\|04\|01\|08\|07\|05\|06 |
| acv_case_04.xlsx | 01       | 1    | 8        | 1     | 01\|04\|02\|03\|05\|06\|07\|08 |
| acv_case_05.xlsx | 04       | 1    | 8        | 1     | 04\|02\|07\|06\|01\|03\|08\|05 |
| acv_case_06.xlsx | 06       | 1    | 8        | 1     | 06\|08\|04\|05\|02\|03\|07\|01 |

**Mean rank-decay score 1.0000** against a random-permutation baseline of 0.5625.

## 2. Leave-one-file-out cross validation

For each held-out file the group weights are re-selected by coordinate ascent on the other five files only, so the held-out score never sees its own tuning.

| held_out         | true_car | rank | score | train_score | ranked_cars                    |
|------------------|----------|------|-------|-------------|--------------------------------|
| acv_case_01.xlsx | 01       | 1    | 1     | 1           | 01\|03\|04\|02\|07\|08\|06\|05 |
| acv_case_02.xlsx | 02       | 1    | 1     | 1           | 02\|03\|08\|07\|06\|04\|01\|05 |
| acv_case_03.xlsx | 03       | 1    | 1     | 1           | 03\|02\|04\|01\|08\|07\|05\|06 |
| acv_case_04.xlsx | 01       | 1    | 1     | 1           | 01\|04\|02\|03\|05\|06\|07\|08 |
| acv_case_05.xlsx | 04       | 1    | 1     | 1           | 04\|02\|07\|06\|01\|03\|08\|05 |
| acv_case_06.xlsx | 06       | 1    | 1     | 1           | 06\|08\|04\|05\|02\|03\|07\|01 |

**LOOCV mean rank-decay score 1.0000.**

## 3. Feature-group ablation

`removed = X` zeroes group X; `all but X` keeps only group X.

| removed             | score | delta  |
|---------------------|-------|--------|
| (nothing)           | 1     | 0      |
| thermal             | 1     | 0      |
| capacity            | 1     | 0      |
| progression         | 1     | 0      |
| control             | 1     | 0      |
| refrigerant         | 0.979 | -0.021 |
| integrity           | 1     | 0      |
| all but thermal     | 0.979 | -0.021 |
| all but capacity    | 0.958 | -0.042 |
| all but progression | 0.812 | -0.188 |
| all but control     | 0.771 | -0.229 |
| all but refrigerant | 0.771 | -0.229 |
| all but integrity   | 0.833 | -0.167 |

## 4. Single-channel discrimination

Rank-decay score each feature achieves on its own. `n_files` is how many files the channel was computable and non-degenerate in.

| group       | feature              | mean_score | n_files | n_top1 | mean_rank | mean_margin |
|-------------|----------------------|------------|---------|--------|-----------|-------------|
| capacity    | greybox_cool_rate    | 1          | 1       | 1      | 1         | 4           |
| capacity    | capacity_shortfall   | 0.979      | 6       | 5      | 1.167     | 3.286       |
| capacity    | load_sensitivity     | 0.875      | 6       | 3      | 2         | 1.83        |
| capacity    | pulldown_rate        | 0.75       | 6       | 3      | 3         | 0.548       |
| control     | full_demand_frac     | 0.531      | 4       | 0      | 4.75      | -1.241      |
| integrity   | integrity_loss       | 1          | 3       | 3      | 1         | 3.349       |
| progression | cusum_fraction       | 0.938      | 6       | 4      | 1.5       | 2.651       |
| progression | elev_trend           | 0.667      | 6       | 3      | 3.667     | 0.628       |
| refrigerant | circuit_asym_lift    | 1          | 1       | 1      | 1         | 1.742       |
| refrigerant | circuit_asym_ratio   | 1          | 1       | 1      | 1         | 4           |
| refrigerant | compressor_cycling   | 1          | 1       | 1      | 1         | 4           |
| refrigerant | lift_deficit         | 1          | 1       | 1      | 1         | 1.019       |
| refrigerant | suction_excursion    | 1          | 1       | 1      | 1         | 4           |
| refrigerant | circuit_asym_high    | 0.75       | 1       | 0      | 3         | -0.644      |
| refrigerant | compressor_duty      | 0.625      | 1       | 0      | 4         | -1.403      |
| thermal     | elev_load_stratified | 0.979      | 6       | 5      | 1.167     | 2.871       |
| thermal     | elev_mean            | 0.979      | 6       | 5      | 1.167     | 2.851       |
| thermal     | elev_persistence     | 0.979      | 6       | 5      | 1.167     | 3.021       |
| thermal     | setpoint_error       | 0.979      | 6       | 5      | 1.167     | 2.981       |
| thermal     | elev_p90             | 0.917      | 6       | 4      | 1.667     | 1.946       |

Rank of the true faulty car per file, by channel:

| group       | feature              | acv_case_01.xlsx | acv_case_02.xlsx | acv_case_03.xlsx | acv_case_04.xlsx | acv_case_05.xlsx | acv_case_06.xlsx |
|-------------|----------------------|------------------|------------------|------------------|------------------|------------------|------------------|
| capacity    | capacity_shortfall   | 1                | 1                | 1                | 2                | 1                | 1                |
| capacity    | greybox_cool_rate    |                  |                  |                  |                  |                  | 1                |
| capacity    | load_sensitivity     | 1                | 2                | 1                | 2                | 5                | 1                |
| capacity    | pulldown_rate        | 1                | 8                | 3                | 1                | 4                | 1                |
| control     | full_demand_frac     | 2                |                  |                  | 4                | 6                | 7                |
| integrity   | integrity_loss       | 1                | 1                | 1                |                  |                  |                  |
| progression | cusum_fraction       | 1                | 1                | 3                | 1                | 2                | 1                |
| progression | elev_trend           | 1                | 1                | 8                | 3                | 1                | 8                |
| refrigerant | circuit_asym_high    |                  |                  |                  | 3                |                  |                  |
| refrigerant | circuit_asym_lift    |                  |                  |                  | 1                |                  |                  |
| refrigerant | circuit_asym_ratio   |                  |                  |                  | 1                |                  |                  |
| refrigerant | compressor_cycling   |                  |                  |                  | 1                |                  |                  |
| refrigerant | compressor_duty      |                  |                  |                  | 4                |                  |                  |
| refrigerant | lift_deficit         |                  |                  |                  | 1                |                  |                  |
| refrigerant | suction_excursion    |                  |                  |                  | 1                |                  |                  |
| thermal     | elev_load_stratified | 1                | 1                | 1                | 2                | 1                | 1                |
| thermal     | elev_mean            | 1                | 1                | 1                | 2                | 1                | 1                |
| thermal     | elev_p90             | 1                | 1                | 1                | 2                | 4                | 1                |
| thermal     | elev_persistence     | 1                | 1                | 1                | 2                | 1                | 1                |
| thermal     | setpoint_error       | 1                | 1                | 1                | 2                | 1                | 1                |

## 5. Calibration on all six files

Coordinate ascent reaches a training score of 1.0000 with group weights `{'thermal': 1.0, 'capacity': 1.0, 'progression': 1.0, 'control': 1.0, 'refrigerant': 1.25, 'integrity': 1.0}`.

Because the physical prior already scores 1.0000 there is nothing for the calibration to correct, and it returns the prior unchanged - the shipped model is therefore the physics, not a fit to six files.

## 6. Block-bootstrap stability

Each case is re-ranked on random 70% subsets of its 6-hour blocks. `top1_true_car` is how often the true faulty car still comes first.

| file_id          | true_car | draws | top1_true_car | most_frequent | most_frequent_freq | mean_rank_true |
|------------------|----------|-------|---------------|---------------|--------------------|----------------|
| acv_case_01.xlsx | 01       | 40    | 1             | 01            | 1                  | 1              |
| acv_case_02.xlsx | 02       | 40    | 1             | 02            | 1                  | 1              |
| acv_case_03.xlsx | 03       | 40    | 1             | 03            | 1                  | 1              |
| acv_case_04.xlsx | 01       | 40    | 0.775         | 01            | 0.775              | 1.23           |
| acv_case_05.xlsx | 04       | 40    | 0.9           | 04            | 0.9                | 1.1            |
| acv_case_06.xlsx | 06       | 40    | 1             | 06            | 1                  | 1              |

This measures *temporal* stability only: every draw still judges each car against the same seven siblings. The companion car-level test, which resamples the consist itself, is test D in `reports/robustness.md`.

## 7. Peer-dropout (car-level) stability

Each case re-ranked from scratch after dropping randomly chosen non-leading cars, with the peer reference, ambient estimate and load strata all recomputed. This tests whether the verdict is a property of the accused car or of the particular sibling group it was compared against.

| file_id          | true_car | draws | cars_dropped | top1_true_car | leader_retained |
|------------------|----------|-------|--------------|---------------|-----------------|
| acv_case_01.xlsx | 01       | 20    | 2            | 1             | 1               |
| acv_case_02.xlsx | 02       | 20    | 2            | 1             | 1               |
| acv_case_03.xlsx | 03       | 20    | 2            | 1             | 1               |
| acv_case_04.xlsx | 01       | 20    | 1            | 0.75          | 0.75            |
| acv_case_05.xlsx | 04       | 20    | 2            | 1             | 1               |
| acv_case_06.xlsx | 06       | 20    | 2            | 1             | 1               |

## 8. Calibrated confidence, not margin

The system used to report the raw top-1 margin as though it were confidence. Test B in `reports/robustness.md` showed that is wrong: deleting the labelled faulty car and re-ranking the healthy siblings produces a margin of the same size, because a peer-consensus ranker always crowns the most anomalous car whether or not one is broken. Confidence is therefore reported as the scale-free Dixon-Q separation referenced to a null of 41 fault-free consists.

| file_id          | true_car | top_car | correct | n_cars | margin | dixon_q | top_z  | elev_mean_top | null_p_value |
|------------------|----------|---------|---------|--------|--------|---------|--------|---------------|--------------|
| acv_case_01.xlsx | 01       | 01      | True    | 8      | 2.035  | 0.499   | 12.406 | 0.479         | 0.214        |
| acv_case_02.xlsx | 02       | 02      | True    | 8      | 1.145  | 0.229   | 10.095 | 0.254         | 0.786        |
| acv_case_03.xlsx | 03       | 03      | True    | 8      | 2.142  | 0.578   | 3.909  | 0.626         | 0.071        |
| acv_case_04.xlsx | 01       | 01      | True    | 4      | 0.095  | 0.057   | 1.979  | 0.224         | 0.952        |
| acv_case_05.xlsx | 04       | 04      | True    | 8      | 0.874  | 0.398   | 2.952  | 0.205         | 0.524        |
| acv_case_06.xlsx | 06       | 06      | True    | 8      | 2.409  | 0.556   | 6.011  | 1.363         | 0.095        |

`null_p_value` is the fraction of fault-free consists separating at least as cleanly. Three of the six genuine faults do not separate distinguishably from a healthy consist, which is the honest state of the evidence rather than a defect: a leak whose capacity deficit is still small genuinely does look like a healthy pack. The metric is reported so the system can say so instead of quoting a confident margin it cannot justify.
