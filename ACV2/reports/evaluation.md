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
| acv_case_06.xlsx | 06       | 1    | 8        | 1     | 06\|08\|04\|05\|02\|03\|01\|07 |

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
| acv_case_06.xlsx | 06       | 1    | 1     | 1           | 06\|08\|04\|05\|02\|03\|01\|07 |

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
| all but progression | 0.792 | -0.208 |
| all but control     | 0.646 | -0.354 |
| all but refrigerant | 0.979 | -0.021 |
| all but integrity   | 0.958 | -0.042 |

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
| acv_case_04.xlsx | 01       | 40    | 0.8           | 01            | 0.8                | 1.2            |
| acv_case_05.xlsx | 04       | 40    | 0.8           | 04            | 0.8                | 1.2            |
| acv_case_06.xlsx | 06       | 40    | 1             | 06            | 1                  | 1              |
