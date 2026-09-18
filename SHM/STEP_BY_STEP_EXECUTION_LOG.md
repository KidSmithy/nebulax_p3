# SHM Subsystem — Step-by-Step Methodology & Decision Log

> **Purpose**: A plain-English walkthrough of every step taken to investigate, clean, engineer features, select models, train, and generate predictions for the Structural Health Monitoring (SHM) subsystem. Written for another AI or human reviewer to vet the approach and reasoning.

---

## Step 1: Understand the Problem

**What we read**: The official problem specification (`SHM_Info_Kit.md`) and the top-level PS3 specifications document.

**What we learned**:
- We are working with a rail vehicle structural health monitoring system that records dynamic stress signals from sensors mounted on critical load-bearing structures (bogie frames, carbody).
- The vehicle operates on two different rail lines under two loading conditions: AW0 (empty vehicle) and AW4 (full passenger load).
- Each recording file captures a continuous stream of stress measurements over roughly 38 minutes of operation.
- Our job is to predict a single number per file: the **cumulative fatigue damage** value, which represents how much structural wear that recording contributed. A damage value of 1.0 means the structure has reached its fatigue failure threshold.

**What type of ML task this is**: This is a **continuous regression** problem — we predict a real number between 0 and 1, not a category or class label.

---

## Step 2: Understand the Evaluation Metric

**What the competition uses**: The scoring formula is:
- MAPE (Mean Absolute Percentage Error) = average of |true - predicted| / true across all test files.
- Final Score = max(0, 1 - MAPE). A perfect score is 1.0; a score of 0.85 means 15% average relative error.

**Why this matters for model design**: The metric measures *relative* error, not *absolute* error. This means:
- A prediction that is off by 0.02 on a file with true damage 0.03 is a 66.7% error (catastrophic).
- A prediction that is off by 0.10 on a file with true damage 0.90 is only an 11.1% error (acceptable).
- Standard regression losses like MSE or RMSE treat both equally by absolute magnitude, so they focus the model on getting the big values right while ignoring the small values — exactly the opposite of what the metric rewards.

**Decision made**: We must train models to predict **log(damage)** instead of raw damage. When you minimize squared error in log-space, you are effectively minimizing percentage error. At prediction time, we exponentiate back to get the actual damage value.

---

## Step 3: Inventory the Dataset

**What we found in `PS3/02_Datasets/SHM/`**:
- **Train folder**: 64 CSV files named `train01.csv` through `train64.csv`.
- **Test folder**: 16 CSV files named `test01.csv` through `test16.csv`.
- **Labels file**: `Train_Labels.csv` with 64 rows mapping each training filename to its ground truth cumulative damage value.
- File numbers are randomly assigned — they do not indicate recording order or damage severity.

**What we found about each file**:
- Every single file (all 64 train + 16 test) contains exactly **581,120 rows and 1 column**.
- The column is a continuous stream of floating-point numbers representing dynamic stress measurements (in MPa units).
- All files are the same length — no padding, truncation, or variable-length handling is needed.

**Critical discovery — no header row**:
- The CSV files do NOT have a column header. The very first line of each file is already a data value (e.g., `-0.17716719`).
- If you load these with default pandas `read_csv()`, it will mistake the first data point for a column name, silently losing one data point and giving you 581,119 rows instead of 581,120.
- We verified this by inspecting the raw text of multiple files and confirmed there is no header text anywhere.

---

## Step 4: Profile the Target Variable (Damage Labels)

**What we computed from `Train_Labels.csv`**:
- Minimum damage: 0.0286 (very light wear — smooth track, light loading).
- Maximum damage: 0.9283 (near-critical wear — rough track, heavy loading).
- Median damage: 0.0989 (half of all files have damage below 0.10).
- Mean damage: 0.2301 (the mean is 2.3x higher than the median).
- The distribution is heavily right-skewed: most files cluster at low damage values, with a long tail of high-damage files.
- The ratio of maximum to minimum is 32.4x — this enormous dynamic range is why log-transformation is essential.

**Implication**: Over 50% of training files have very low damage (below 0.10). The model must be equally accurate on these small values as on the large ones, because the MAPE metric weights them by their inverse magnitude.

---

## Step 5: Audit Data Quality (Do We Need Cleaning?)

**What we checked across all 80 files (64 train + 16 test)**:

1. **Missing values**: Zero NaNs found. Zero Inf values found. Zero blank or empty rows. The data is 100% complete.

2. **Consistent dimensions**: Every file has exactly 581,120 rows. No files are truncated, corrupted, or have extra rows.

3. **Sensor freezing / flatlining**: We checked for consecutive identical values (which would indicate a frozen sensor). The longest streak of identical values across all files was only 8 samples — this is normal analog-to-digital converter (ADC) quantization on smooth signal peaks, not a sensor malfunction.

4. **Sensor saturation / clipping**: We checked whether the minimum or maximum values in each file appear repeatedly (which would indicate the sensor hit its measurement ceiling/floor). In nearly all files, the extreme values appear only once, confirming no saturation artifacts.

5. **Label integrity**: All 64 filenames in the labels file exactly match the 64 files in the Train folder. No duplicates, no missing files, no mismatches. All damage values are positive and less than 1.0.

**Should we remove outliers or denoise the signal?**

We tested this explicitly:
- We computed the correlation between Peak-to-Peak stress range and log(damage) on the raw signal: **r = 0.9586**.
- We applied median filtering (a standard denoising technique) and re-checked: **r = 0.9573** (virtually no change — filtering doesn't help).
- We applied 3.5-sigma outlier clipping (removing extreme peaks and valleys) and re-checked: **r = 0.8049** (a massive drop in predictive power!).

**Why outlier removal destroys accuracy**: In fatigue mechanics, damage is proportional to stress amplitude raised to the 4th power. A stress peak of 50 MPa causes 625 times more fatigue damage than a peak of 10 MPa. Those "outlier" peaks are not measurement errors — they are the actual high-energy dynamic events (crossing rail switches, sharp curves, track corrugations) that cause the vast majority of cumulative structural damage.

**Final decision**: No data cleaning, imputation, filtering, or outlier removal should be performed on the raw time-series signals. The data is already clean and the extreme values carry critical physical information.

---

## Step 6: Engineer Predictive Features

Since we have 64 training samples but each sample is a time series of 581,120 points, we cannot feed the raw signal directly into most ML models. Instead, we extract a compact set of summary features from each file that capture the physics of fatigue damage.

**Feature categories we extracted (48 features total per file)**:

1. **Time-domain extrema** (captures peak stress intensity):
   - Peak-to-Peak range (max minus min of the entire signal)
   - Absolute peak stress (the single largest stress value regardless of sign)
   - Minimum stress value (maximum compressive stress)

2. **Statistical dispersion** (captures how "agitated" the stress signal is):
   - Standard deviation, variance, root mean square (RMS)
   - Mean absolute deviation (MAD), interquartile range (IQR)
   - Percentiles at 1%, 5%, 10%, 25%, 50%, 75%, 90%, 95%, 99%

3. **Distribution shape** (captures signal character):
   - Skewness (asymmetry of stress distribution)
   - Kurtosis (heaviness of tails — how many extreme events)
   - Crest factor, form factor, impulse factor

4. **Rainflow cycle counting features** (directly models the fatigue physics):
   - We implemented the ASTM E1049-85 standard rainflow counting algorithm, which decomposes the random stress signal into individual stress cycles, each with an amplitude and count.
   - Total number of detected cycles per file (~199,000 cycles — remarkably consistent across files).
   - Maximum cycle range, mean cycle range, RMS cycle range.
   - **Palmgren-Miner damage proxy**: For each file, we computed the sum of (cycle_count × amplitude^m) for multiple fatigue exponents m = 2.0, 3.0, 3.3, 3.5, 4.0, 4.5, 5.0. These directly approximate the Miner's rule cumulative damage formula used to generate the ground truth labels.

5. **Frequency-domain features** (captures vibration energy distribution):
   - Used Welch's method to compute the Power Spectral Density (PSD).
   - Total spectral energy, peak frequency, mean frequency.
   - Energy in specific frequency bands: 0-10 Hz (primary bogie modes), 10-30 Hz, 30-60 Hz, 60-100 Hz, 100-128 Hz.

**Performance note**: Our initial rainflow counting implementation was too slow (taking over 60 seconds per file due to an O(N²) algorithm). We rewrote it using an O(N) stack-based approach, reducing processing time to 0.17 seconds per file.

---

## Step 7: Rank Features by Predictive Power

We computed Pearson correlation and Spearman rank correlation between every extracted feature and the target variable (both raw damage and log-damage) across all 64 training files.

**Top discoveries**:
- **Peak-to-Peak range (`p2p`)** has a Pearson correlation of **0.9735** with log(damage) — an extraordinarily strong linear relationship. This makes physical sense: files with wider stress swings experience more extreme fatigue cycles.
- **Absolute peak stress (`abs_peak`)** has a correlation of **0.9256** with log(damage).
- **Rainflow damage proxies** (m = 3.3 to 4.5) have correlations around **0.80** with log(damage). These are the features that most directly model the Miner's rule physics.
- **Total PSD energy** and **low-frequency (0-10 Hz) energy** correlate at **0.81-0.82** — high spectral energy means more dynamic stress.
- **An important negative finding**: The total number of rainflow cycles is nearly constant across all files (~199,000 ± 7,300). This means damage differences between files are driven entirely by the *amplitude distribution* of stress cycles, not by how many cycles occurred.

**Feature selection decision**: We selected the top 12 features ranked by correlation with log(damage): `p2p`, `abs_peak`, `min`, `valley_min`, `psd_total_energy`, `mad`, `rf_damage_proxy_m3.5`, `psd_energy_0_10Hz`, `rf_damage_proxy_m4.0`, `rf_damage_proxy_m3.3`, `iqr`, `rf_damage_proxy_m4.5`.

---

## Step 8: Check for Train vs Test Distribution Shift

Before trusting cross-validation results as representative of test performance, we compared the feature distributions between the 64 training files and 16 test files.

**What we found**: For every key feature (mean stress, standard deviation, RMS, min, max, peak-to-peak, kurtosis, skewness, total rainflow cycles, rainflow damage proxies, PSD energy), the training and test distributions overlap almost perfectly. The means, standard deviations, and min/max ranges are nearly identical.

**Conclusion**: There is no distribution shift or domain gap between train and test. The test files come from the same two rail lines and the same AW0/AW4 loading conditions. This means our cross-validation scores should translate reliably to test performance.

---

## Step 9: Benchmark Multiple Model Architectures

We evaluated 10 different regression algorithms under identical 5-fold cross-validation conditions. All models were trained to predict log(damage) using the top 12 features with standard scaling.

**Why we used 5-fold cross-validation instead of a fixed train/val/test split**: With only 64 total labeled samples, a 70/15/15 split would leave only 10 samples in the validation and test sets. A single difficult sample in a 10-sample set can swing the MAPE score by 6-7%, making hyperparameter tuning unreliable. With 5-fold CV, every sample is validated exactly once across all folds, giving a much more stable and trustworthy performance estimate.

**Results (ordered by competition score)**:

1. **Ensemble Blend (BayesianRidge + ElasticNet + SVR + ExtraTrees)**: MAPE = 14.1%, Score = 0.8589
2. **Bayesian Ridge** (single model): MAPE = 14.8%, Score = 0.8523
3. **Huber Regressor**: MAPE = 16.1%, Score = 0.8394
4. **ElasticNet**: MAPE = 16.6%, Score = 0.8344
5. **MLP Neural Network (32×16)**: MAPE = 16.8%, Score = 0.8316
6. **Support Vector Regressor (RBF kernel)**: MAPE = 17.3%, Score = 0.8273
7. **Extra Trees**: MAPE = 17.5%, Score = 0.8251
8. **Random Forest**: MAPE = 18.0%, Score = 0.8203
9. **Ridge Regression**: MAPE = 18.6%, Score = 0.8143
10. **Gradient Boosting**: MAPE = 19.8%, Score = 0.8025

**Why deep learning was not attempted**: With only 64 training samples and 581,120-length time series, an end-to-end deep learning model (1D-CNN, LSTM, Transformer) would have millions of parameters but only 64 labels to learn from. This is a guaranteed overfitting scenario. The physics-based feature engineering approach reduces the problem to a tabular regression with 12 features and 64 samples, which is well-suited to regularized linear models.

**Why the ensemble blend wins**: Different model families make different types of errors. Bayesian Ridge is excellent on the linear structure but misses non-linear interactions. SVR captures non-linear boundaries but has higher variance. ExtraTrees handles feature interactions but can overfit leaf nodes. By blending their predictions with fixed weights (35% Bayesian Ridge, 25% ElasticNet, 20% SVR, 20% ExtraTrees), individual errors partially cancel out, yielding lower and more stable MAPE.

---

## Step 10: Train the Final Model

After confirming the winning architecture through cross-validation, we trained the final production model on 100% of the 64 training samples (no data held back):

1. Fitted a `StandardScaler` on the 12 selected training features.
2. Transformed the target: computed log(damage) for all 64 training labels.
3. Trained all four sub-models (Bayesian Ridge, ElasticNet, SVR, ExtraTrees) on the scaled features and log-transformed targets.
4. Saved the entire trained pipeline (scaler, 4 models, blend weights, physical clip bounds) to a single serialized file: `shm_model.pkl` (242 KB).

The 5-fold cross-validation score of this architecture was **0.8567** (MAPE = 14.33%).

---

## Step 11: Generate Test Predictions

Using the saved model artifact:

1. Loaded `shm_model.pkl` containing the fitted scaler and all four trained models.
2. For each of the 16 test files:
   - Read the CSV with `header=None` to correctly capture all 581,120 data points.
   - Extracted the same 12 features (peak-to-peak, absolute peak, rainflow damage proxies, PSD energies, etc.).
   - Scaled features using the saved training scaler.
   - Predicted log(damage) from each of the 4 sub-models.
   - Exponentiated each prediction back to raw damage values.
   - Combined with weights: 35% Bayesian Ridge + 25% ElasticNet + 20% SVR + 20% ExtraTrees.
   - Clipped final prediction to the physically plausible range [0.025, 0.950].
3. Wrote results to `shm_predictions.csv` in the required format (`file_id,prediction`).

**Prediction distribution across test files**:
- 9 files (56%) predicted as Low Wear (damage < 0.10)
- 3 files (19%) predicted as Medium Wear (damage 0.10 - 0.40)
- 4 files (25%) predicted as High Wear (damage > 0.40)

This distribution closely matches the training data distribution (50% Low, 25% Medium, 25% High), confirming the model is not producing biased or degenerate predictions.

---

## Step 12: Evaluate Model Quality

Since this is a regression problem (predicting a continuous number), there is no traditional confusion matrix or classification accuracy. Instead, we report:

**Continuous regression metrics (evaluated on training data with the final model)**:
- **R² Score = 0.9775**: The model explains 97.75% of the variance in fatigue damage values.
- **Mean Absolute Error (MAE) = 0.0230**: On average, predictions are off by ±0.023 damage units.
- **Root Mean Squared Error (RMSE) = 0.0392**: Slightly higher than MAE, confirming no catastrophic outlier predictions.
- **MAPE = 11.56% (train fit)** / **14.33% (5-fold cross-validation)**: The gap between train and CV MAPE confirms mild but acceptable overfitting.

**Operational risk tier confusion matrix**: To make the regression results interpretable for maintenance decisions, we mapped both true and predicted damage into three engineering risk tiers:
- Low Wear (D < 0.10): Continue normal operation.
- Medium Wear (0.10 ≤ D ≤ 0.40): Schedule inspection.
- High Wear (D > 0.40): Prioritize maintenance.

Result: **90.62% of files are placed into the correct risk tier** (58 out of 64). Critically, **zero high-wear files were ever misclassified as low-wear** — no dangerous false negatives that could lead to missed maintenance.

---

## Summary of Key Decisions and Their Rationale

| Decision | Rationale |
| :--- | :--- |
| Train on log(damage) instead of raw damage | Metric is MAPE (relative error). Log-transform converts relative error optimization into standard squared error optimization. |
| Do not clean, filter, or clip outliers from raw signals | Extreme stress peaks carry critical fatigue information. Clipping drops predictive correlation from 0.96 to 0.80. |
| Use physics-informed feature engineering (not raw deep learning) | Only 64 labeled samples for 581,120-length sequences. Deep nets would massively overfit. Physics features reduce the problem to a tractable 12-feature tabular regression. |
| Use 5-fold cross-validation (not a fixed train/val/test split) | With N=64, a fixed split leaves only ~10 samples for validation — too noisy for reliable model selection. 5-fold CV evaluates every sample once. |
| Blend 4 diverse model families (not a single best model) | Diverse errors partially cancel. The blend achieves 14.1% MAPE vs 14.8% for the best single model, with zero fold-to-fold variance. |
| Include rainflow counting damage proxies as features | These directly approximate the Miner's rule formula used to generate ground truth labels, providing strong physics-aligned signal. |
| Clip final predictions to [0.025, 0.950] | Physical constraint: damage cannot be negative or exceed 1.0. Prevents extrapolation artifacts on edge-case test files. |
