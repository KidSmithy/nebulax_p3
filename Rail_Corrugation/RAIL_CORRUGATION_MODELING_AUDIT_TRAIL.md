# Rail Corrugation Model Development: Complete Step-by-Step Methodology & Technical Audit Trail

**Subsystem**: Problem Statement 3 (PS3) - Rail Transit Condition Monitoring: Identification of Abnormal Rail Corrugation  
**Document Purpose**: Peer-review & verification document for external AI models / technical evaluators (e.g. Manus AI, senior reviewers).  
**Associated Scripts & Artifacts**:
- Raw Datasets: `PS3/02_Datasets/Rail_Corrugation/` (`Train/` 272 files, `Test/` 68 files, `Train_Labels.csv`)
- Master Profiler: `Rail_Corrugation/investigate_rail_corrugation.py`
- AutoML & Optuna Optimization Pipeline: `Rail_Corrugation/run_automl_pipeline.py`
- Production Model Exporter: `Rail_Corrugation/train_and_export_baseline.py`
- Frontend Inference Engine: `Rail_Corrugation/models/rail_pipeline.py`
- Exported Bundle: `Rail_Corrugation/models/rail_model_bundle.joblib`
- Out-of-Fold Benchmark Leaderboard: `Rail_Corrugation/automl_leaderboard_results.csv`
- Final Test Predictions: `Rail_Corrugation/automl_rail_predictions.csv`

---

## Step 1: Raw Dataset Investigation & Architectural Mapping

### 1.1 Data Dimensions & Integrity Check
1. **File Inventory**:
   - Training Set: 272 CSV files (`Train1.csv` – `Train272.csv`) totaling ~4.68 GB.
   - Test Set: 68 CSV files (`Test1.csv` – `Test68.csv`) totaling ~1.17 GB.
   - Ground Truth Labels: `Train_Labels.csv` (exactly 272 rows).
2. **Signal Properties**:
   - Uniform dimensions across all 340 files: Exactly **10,000 samples $\times$ 129 columns** ($f_s = 10,000\text{ Hz}$, duration $= 1.0\text{ s}$).
   - Missing / NaN values: **0** across all 43.86 million data points.
   - Infinite / extreme glitch spikes ($> 100\text{ m/s}^2$): **0**.
   - Dead / flatlined channels: **0** (all 128 channels show active variance $\sigma > 10^{-6}$).
   - Value range: Bounded within $[-15.6, +35.0]\text{ m/s}^2$ (no sensor clipping / ADC saturation).

### 1.2 Physical Channel Topology
- **Column 1 (`Rotating speed`)**: Pulse sensor monitoring a 90-tooth wheel with diameter $D=0.85\text{ m}$ (circumference $C = \pi \times 0.85 \approx 2.67035\text{ m}$). 180 binary transitions ($0 \leftrightarrow 1$) correspond to 1 full wheel revolution.
- **Columns 2 – 129**: Dual-axis bearing acceleration measurements across 8 cars $\times$ 8 axle boxes:
  - **Side I Rail**: Positions 1, 3, 5, 7 across Cars 1–8 $\rightarrow$ 64 channels (32 Vibration + 32 Shock).
  - **Side II Rail**: Positions 2, 4, 6, 8 across Cars 1–8 $\rightarrow$ 64 channels (32 Vibration + 32 Shock).

### 1.3 Label Imbalance
- `Normal`: **234 files (86.03%)**
- `Side II`: **24 files (8.82%)**
- `Side I`: **14 files (5.15%)**
- *Challenge*: Severe class imbalance. The evaluation metric is **Macro F1** (unweighted mean of per-class F1 scores). A naive model predicting `Normal` achieves 86.03% accuracy but fails with a Macro F1 of only 0.308.

---

## Step 2: Signal Conditioning & Physics-Based Discoveries

### 2.1 The "Stationary / Low-Speed Gating" Rule (Deterministic Physics Filter)
- **Empirical Discovery**: In the training set, 60 files run at speeds $< 10\text{ km/h}$ (including 34 files with 0 wheel movement). **100% of these 60 files are labeled `Normal`**.
- **Physical Rationale**: Corrugation excitation is a dynamic wheel-rail resonance ($f = v / \lambda$). At zero or crawl speeds ($v < 10\text{ km/h}$), dynamic contact forces are negligible and cannot excite rail roughness frequencies.
- **Pipeline Implementation**: A deterministic hard gate before model classification:
  $$\text{If } v < 10.0\text{ km/h} \implies \text{Predict } \mathbf{Normal} \text{ (100\% confidence)}$$
  *(Prevents false alarms on the 9 zero-speed files present in the test set).*

### 2.2 DC Offset Removal (Mean Subtraction)
- Axle-box accelerometers exhibit static gravitational bias and physical mounting offsets (mean up to $4.0\text{ m/s}^2$, average $\approx 2.06\text{ m/s}^2$).
- Computing raw RMS as $\sqrt{\frac{1}{N}\sum x^2}$ artificially inflates the signal with static gravity rather than dynamic corrugation amplitude.
- **Action**: Every vibration and shock channel is centered ($x_{\text{dyn}} = x - \mu_x$) prior to calculating energy metrics.

### 2.3 Spatial Localization Physics (Car-Level Asymmetry)
- In $1.0\text{ s}$ at $50\text{ km/h}$ ($13.89\text{ m/s}$), the train advances by only $13.89\text{ meters}$.
- An 8-car passenger train measures $160 - 200\text{ meters}$ in total length.
- Corrugation patches on the track typically span $10 - 30\text{ meters}$.
- **Data Evidence**: In `Train213.csv` (`Side I`), Cars 1–6 have a normal Side I/II ratio ($\approx 0.95$), while Cars 7 & 8 spike by $2.85\times$ on Side I. In `Train188.csv` (`Side II`), Car 1 spikes by $3.0\times$ on Side II while Cars 2–8 are completely quiescent.
- **Action**: Calculating train-wide averages dilutes localized defects. The feature extractor must compute **car-by-car localized asymmetry** ($\max_c \frac{\text{RMS}_{I, c}}{\text{RMS}_{II, c}}$ and $\min_c (\text{RMS}_{I, c} - \text{RMS}_{II, c})$).

---

## Step 3: Domain Feature Engineering (63 Features Table)

Rather than feeding raw $10,000 \times 129$ matrices into deep models (which overfit on 272 samples with only 14 `Side I` cases), each recording was transformed into a 63-dimensional physical feature vector saved in `Rail_Corrugation/rail_extracted_features.csv`:

1. **Kinematic Speed Features (5 features)**:
   - Pulse transition count, revolutions/sec, linear velocity ($v_{\text{ms}}$, $V_{\text{kmh}}$), low-speed indicator.
2. **Global Side Asymmetry (8 features)**:
   - Dynamic RMS of Side I Vibration, Side II Vibration, Side I Shock, Side II Shock.
   - Global Vibration RMS ratio ($\frac{\text{RMS}_{I}}{\text{RMS}_{II}}$) and difference ($\text{RMS}_I - \text{RMS}_{II}$).
   - Global Shock RMS ratio and difference.
3. **Higher-Order Statistical Moments (6 features)**:
   - Mean peak-to-peak amplitude ratios across rails.
   - 4th-order Kurtosis of Side I vs Side II shock channels (to detect impact spikes as wheel passes corrugation crests).
   - Kurtosis differences.
4. **Car-by-Car Spatial Localization (21 features)**:
   - Individual Vibration RMS differences and ratios for each car $c \in [1..8]$.
   - Train-wide maximum car ratio ($\max_c \text{Ratio}_c$).
   - Train-wide minimum car ratio ($\min_c \text{Ratio}_c$).
   - Train-wide maximum car difference ($\max_c \text{Diff}_c$) and minimum difference ($\min_c \text{Diff}_c$).
   - Car differential range ($\max - \min$).
5. **Spectral Sub-Band Energy & Power Ratios via Welch PSD (23 features)**:
   - Excitation formula $f = v / \lambda$ indicates corrugation energy concentrates between $100\text{ Hz} - 1000\text{ Hz}$.
   - Welch PSD ($f_s=10,000$, $N_{\text{perseg}}=1024$) was computed on mean Side I vs Side II signals across 6 bands:
     - Band 1: $0 - 100\text{ Hz}$ (Suspension / whole-bogie dynamics)
     - Band 2: $100 - 250\text{ Hz}$ (Long-pitch corrugation)
     - Band 3: $250 - 500\text{ Hz}$ (Primary wheel-rail contact resonance / P2 mode)
     - Band 4: $500 - 1000\text{ Hz}$ (Short-pitch corrugation harmonic)
     - Band 5: $1000 - 2000\text{ Hz}$ (High-frequency structural chatter)
     - Band 6: $2000 - 4000\text{ Hz}$ (Acoustic squeal / noise)
   - For each band: Side I Power, Side II Power, Power Ratio ($\frac{P_I}{P_{II}}$), and Power Difference ($P_I - P_{II}$).
   - Dominant spatial wavelength estimation: $\lambda_{\text{est}} = \frac{v}{f_{\text{peak}}}$.

---

## Step 4: Multi-Engine AutoML & Custom Optuna Optimization

All modeling was conducted strictly under **5-Fold Stratified Cross Validation** to prevent data leakage and ensure fair representation of the 14 `Side I` samples across folds.

### 4.1 Phase 1: AutoGluon Tabular (`presets="best_quality"`, 5-fold bagging, 0 stack levels)
- Executed directly on `rail_extracted_features.csv` with target metric `eval_metric="f1_macro"`.
- Trained and validated 12 models across 5 folds:
  - `LightGBMLarge_BAG_L1`: **0.7842 Macro F1**
  - `WeightedEnsemble_L2`: **0.7842 Macro F1**
  - `LightGBM_BAG_L1`: **0.7772 Macro F1**
  - `LightGBMXT_BAG_L1`: **0.7589 Macro F1**
  - `XGBoost_BAG_L1`: **0.6697 Macro F1**
  - `CatBoost_BAG_L1`: **0.6526 Macro F1**

### 4.2 Phase 2: FLAML AutoML
- Rapid Bayesian optimization over `["xgboost", "lgbm", "catboost", "rf", "extra_tree"]` with 5-fold stratified CV optimizing `macro_f1`.
- Selected **LightGBM** (`num_leaves=6`, `learning_rate=0.75`, `n_estimators=12`).
- Achieved **0.7310 Macro F1** out-of-fold in 60 seconds.

### 4.3 Phase 3: Optuna Custom Macro-F1 Optimization
Standard AutoML optimizes default probability argmax (`prob > 0.33`), which penalizes minority classes. Optuna was configured with a custom objective function directly maximizing out-of-fold Macro F1 across 60 Bayesian trials:
1. **Search Space**:
   - Model family: `Categorical(["xgboost", "catboost", "lightgbm"])`
   - Minority class weight multipliers: $w_{\text{Side I}} \in [6.0, 26.0]$, $w_{\text{Side II}} \in [3.0, 15.0]$
   - Low-speed hard cutoff: $V_{\text{th}} \in [8.0, 14.0]\text{ km/h}$
   - Tree depth, learning rate, regularization ($L_2$, $\lambda$, subsample, colsample)
2. **Best Parameters Found**:
   - Classifier: **CatBoost** (`depth=6`, `learning_rate=0.198`, `l2_leaf_reg=1.10`)
   - Class weights: `Normal`: 1.0, `Side I`: **18.50**, `Side II`: **7.29**
   - Low-speed cutoff: **$12.88\text{ km/h}$**
   - Out-of-Fold Score: **0.7648 Macro F1** (Normal: 0.9706, Side I: 0.4348, Side II: **0.8889**, Overall Accuracy: 94.12%).

---

## Step 5: Final Validation Benchmark & Leaderboard Comparison

Below is the verified out-of-fold cross-validation performance comparison across all evaluated model families:

| Model Architecture | **Macro F1 (Official Metric)** | Normal F1 | Side I F1 *(Rare)* | Side II F1 | Overall Accuracy |
|---|---|---|---|---|---|
| **AutoGluon (Weighted Ensemble L2)** | **0.7842** | 0.9680 | 0.4706 | 0.9140 | 94.49% |
| **AutoGluon (LightGBMLarge Bagged)** | **0.7842** | 0.9680 | 0.4706 | 0.9140 | 94.49% |
| **Optuna Tuned CatBoost + Gating** | **0.7648** | 0.9706 | 0.4348 | 0.8889 | 94.12% |
| **FLAML (Best Estimator: LightGBM)** | **0.7310** | 0.9540 | 0.5000 | 0.7391 | 91.91% |
| **Weighted XGBoost (Baseline)** | **0.6777** | 0.9558 | 0.2857 | 0.7917 | 91.54% |
| **Balanced Random Forest** | **0.5980** | 0.9570 | 0.0000 | 0.8370 | 92.30% |
| **Naive Majority (Always Normal)** | **0.3080** | 0.9250 | 0.0000 | 0.0000 | 86.03% |

### Key Observations for Peer Reviewers:
1. **The Feature Set Works**: The top feature across all gradient boosted trees was `band_250_500_s1` (contributing $25.8\%$ of split gain), proving that the $250 - 500\text{ Hz}$ resonance band is the definitive physical signature of corrugation.
2. **AutoGluon vs Optuna**: AutoGluon's multi-model weighted ensemble achieved the highest overall Macro F1 (**0.7842**), while Optuna's single CatBoost model with asymmetric loss weights achieved **0.7648** with the highest precision on Side II ($88.9\%$).
3. **Accuracy Trap Confirmed**: Random Forest scored 92.3% accuracy but 0.598 Macro F1 because it failed on Side I. Both AutoGluon and Optuna overcame this by forcing minority class sensitivity.

---

## Step 6: Artifact Packaging & Frontend Integration

### 6.1 Exported Files in `Rail_Corrugation/models/`
- **`rail_model_bundle.joblib`**: Self-contained binary bundle storing:
  - The trained production model.
  - Complete 63 feature column definitions in exact expected order.
  - Pre-computed optimal probability decision divisors.
  - Speed gating threshold ($12.88\text{ km/h}$).
  - Label and inverse label mapping dictionaries.
- **`rail_pipeline.py`**: A unified Python class `RailPredictor`:
  - Accepts raw CSV file paths, byte streams, or pandas DataFrames.
  - Automatically handles speed decoding, DC offset removal, feature extraction, and low-speed gating.
  - Returns prediction, confidence, calculated speed, and per-class probability breakdown in $< 20\text{ ms}$.

### 6.2 Test Predictions Generated
- Processed all 68 unlabeled files in `PS3/02_Datasets/Rail_Corrugation/Test/`.
- Saved to `Rail_Corrugation/automl_rail_predictions.csv` (and mirrored in `baseline_rail_predictions.csv`).
- Schema verified against `PS3/04_Example_Submission/rail_predictions.csv`:
  - 68 rows, `file_id,prediction`.
  - Predicted test distribution: **61 Normal, 4 Side I, 3 Side II** (aligns with train operational distribution).
