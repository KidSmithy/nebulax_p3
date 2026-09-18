# Technical Audit & Vetting Guide: Rail Corrugation Condition Monitoring Pipeline

> **Purpose**: This document provides an exhaustive, peer-review-ready breakdown of the machine learning pipeline developed for the Track Rail Corrugation Multi-Class Monitoring Problem. It covers physical principles, data pre-processing, domain feature extraction, cross-validation protocols, model search, threshold optimization, and inference artifacts. Another AI model or data scientist can use this specification to vet the methodology for data leakage, mathematical validity, and metric alignment.

---

## 1. Problem Framing & Dataset Context

### 1.1 Objective
Classify 1-second dynamic sensor recordings from an 8-car train into one of three operational track health states:
- `Normal` (Class 0): Both rails healthy or low-speed non-corrugated segments.
- `Side I` (Class 1): Abnormal corrugation defect on the left/inner rail.
- `Side II` (Class 2): Abnormal corrugation defect on the right/outer rail.

### 1.2 Evaluation Metric
- **Macro-averaged F1 Score**:
  $$\text{Macro F1} = \frac{1}{3} \left( F_{1,\text{Normal}} + F_{1,\text{Side I}} + F_{1,\text{Side II}} \right)$$
- Optimizing for overall accuracy alone creates severe minority starvation due to class imbalance.

### 1.3 Data Dimensions & Class Imbalance
- **Training Set**: 272 CSV recordings (10,000 time steps $\times$ 129 channels, sampled at 10 kHz = 1 second duration).
  - `Normal`: 234 samples (86.03%)
  - `Side II`: 24 samples (8.82%)
  - `Side I`: 14 samples (5.15%) $\rightarrow$ *Primary bottleneck: severe minority class*.
- **Test Set**: 68 unlabelled CSV recordings (same channel structure and duration).
- **Sensors**:
  - `Rotating speed`: Speed pulse sensor on the axle.
  - Channels 1–64: Bearing Vibration across 8 cars (Cars 1–8), 8 positions per car (Positions 1, 3, 5, 7 on Side I; Positions 2, 4, 6, 8 on Side II).
  - Channels 65–128: Bearing Shock across 8 cars, same position/side split.

---

## 2. Physics-Informed Preprocessing & Signal Processing

### 2.1 Axle Speed Decoding & Physical Low-Speed Gate
1. **Speed Extraction**:
   - Pulse transition count $N_{\text{trans}} = \sum_{t=1}^{T-1} \mathbb{I}(|\Delta s_t| > 0)$.
   - Axle revolutions: $\text{Revs} = \frac{N_{\text{trans}}}{180}$.
   - Wheel diameter $D = 0.85\text{ m} \implies \text{Circumference } C = \pi D \approx 2.67035\text{ m}$.
   - Linear speed $v = \text{Revs} \times C\text{ (m/s)} = v \times 3.6\text{ (km/h)}$.
2. **Deterministic Physics Gate (< 10.0 km/h)**:
   - *Physical rationale*: Dynamic rail corrugation excitation requires rolling velocity to induce the 20–1000 Hz wheel-rail dynamic contact resonance. At stationary ($0\text{ km/h}$) or creeping speeds ($< 10\text{ km/h}$), no periodic corrugation frequency is excited.
   - *Empirical ground truth*: In the training set, 100% of samples with $v < 10\text{ km/h}$ are `Normal`.
   - *Inference rule*: If $v < 10.0\text{ km/h}$, output `Normal` with 1.0 confidence without model evaluation. In the test set, 16 out of 68 files (23.5%) fall below 10 km/h.

### 2.2 DC Offset Elimination
Sensor zero-point drifts and gravitational bias are eliminated via channel-wise dynamic centering:
$$\tilde{x}_c(t) = x_c(t) - \frac{1}{T}\sum_{k=1}^T x_c(k)$$

---

## 3. 92-Feature Engineering Taxonomy

Rather than feeding raw $10,000 \times 129$ matrices, domain signal processing collapses each recording into a 92-dimensional feature vector across 5 functional categories:

### Category A: Speed & Kinematics (5 features)
- `transitions`, `speed_ms`, `speed_kmh`, `is_low_speed` ($<15\text{ km/h}$), `is_zero_speed`.

### Category B: Global Left-vs-Right Energy & Impulsiveness (12 features)
- Side I vs Side II Root Mean Square (RMS) for Vibration and Shock:
  $$\text{RMS}_{s} = \sqrt{\frac{1}{N_{\text{channels}} T} \sum_{c \in \text{Side } s} \sum_{t=1}^T \tilde{x}_c(t)^2}$$
- `v_rms_ratio` ($\frac{\text{RMS}_{v,1}}{\text{RMS}_{v,2} + \epsilon}$), `v_rms_diff` ($\text{RMS}_{v,1} - \text{RMS}_{v,2}$).
- `s_rms_ratio`, `s_rms_diff`.
- `v_ptp_ratio` (Peak-to-peak amplitude ratio).
- Shock Kurtosis and ratios: `s1_s_kurt`, `s2_s_kurt`, `s_kurt_ratio`, `s_kurt_diff`.

### Category C: Spatial Localization & Car Order Statistics (34 features)
A train is $\approx 160\text{ m}$ long, but in 1 second at 50 km/h the train advances only $\approx 14\text{ m}$. A corrugated track defect localized to a specific rail stretch primarily excites only 1–3 bogies during the 1-second window.
- Individual car ratios & diffs across Cars 1–8: `car1_v_ratio` to `car8_v_ratio`, `car1_v_diff` to `car8_v_diff`.
- Maximum and minimum car response: `max_car_v_ratio`, `min_car_v_ratio`, `max_car_v_diff`, `min_car_v_diff`, `car_v_diff_range`.
- **Top-order statistics (Asymmetry peaks)**:
  - `top1_car_ratio`, `top2_car_ratio`, `top3_car_ratio`
  - `mean_top2_car_ratio`, `mean_top3_car_ratio`
  - `top1_car_diff`, `top2_car_diff`, `mean_top2_car_diff`
- **Bottom-order statistics (Side II indicators)**:
  - `min1_car_ratio`, `min2_car_ratio`, `mean_min2_car_ratio`, `min1_car_diff`, `min2_car_diff`, `mean_min2_car_diff`.
- Inter-car variance: `car_ratio_std`, `car_ratio_range`.

### Category D: Spectral Welch Power & Centroid/Entropy (27 features)
- Welch Power Spectral Density ($f_s = 10,000\text{ Hz}$, $N_{\text{perseg}} = 1024$):
- Fixed frequency bands:
  - $[0, 100\text{ Hz}]$, $[100, 250\text{ Hz}]$, $[250, 500\text{ Hz}]$, $[500, 1000\text{ Hz}]$, $[1000, 2000\text{ Hz}]$, $[2000, 4000\text{ Hz}]$.
  - For each band: Side I power, Side II power, power ratio ($P_1 / P_2$), power difference ($P_1 - P_2$).
- Spectral Centroid & Entropy Difference:
  - $\text{Centroid} = \frac{\sum f \cdot P(f)}{\sum P(f)}$
  - $\text{Entropy} = -\sum p_i \log_2(p_i)$ where $p_i = \frac{P(f_i)}{\sum P(f)}$
  - `spectral_centroid_diff`, `spectral_entropy_diff`.
- Dominant wavelength: $\lambda_{\text{dom}} = \frac{v}{f_{\text{peak}}}$.

### Category E: Speed-Normalized Spatial Wavelength Bands (14 features)
Rail corrugations have fixed physical wavelengths $\lambda$ regardless of train speed, but observed temporal frequencies shift with speed ($f = \frac{v}{\lambda} \iff \lambda = \frac{v}{f}$).
Dynamic frequency bounds are calculated based on instantaneous velocity $v$:
$$\left[ f_{\text{low}}, f_{\text{high}} \right] = \left[ \frac{v}{\lambda_{\text{max}}}, \frac{v}{\lambda_{\text{min}}} \right]$$
Extracted across 5 canonical railway corrugation wavelength regimes:
- $[2\text{--}4\text{ cm}]$, $[4\text{--}7\text{ cm}]$, $[7\text{--}12\text{ cm}]$, $[12\text{--}20\text{ cm}]$, $[20\text{--}30\text{ cm}]$.
- For each band: power ratio ($P_1 / P_2$) and difference ($P_1 - P_2$).

---

## 4. Cross-Validation & Modeling Methodology

### 4.1 Strict Leak-Free Cross-Validation
- Stratified 5-Fold Cross-Validation: Each fold preserves class balance (ensuring $\approx 2\text{--}3$ Side I examples per validation fold).
- 5-Fold $\times$ 5-Seed Repeated Cross Validation (25 evaluations total) to confirm stability against high variance in rare-class metrics.
- All signal transformations, Welch transforms, and normalizations operate strictly per-file (zero inter-file feature leakage).

### 4.2 Model Exploration & Benchmarking
Multiple model paradigms were tuned directly against the `f1_macro` metric:

| Architecture | Strategy / Hyperparameters | Out-of-Fold Macro F1 | Side I F1 | Side II F1 |
| :--- | :--- | :---: | :---: | :---: |
| **Weighted XGBoost (Baseline)** | `max_depth=4`, `lr=0.08`, inverse sample weights | 0.6777 | 0.2857 | 0.7917 |
| **FLAML AutoML** | Iterative search across RF, ExtraTrees, LGBM, XGB | 0.7310 | 0.5000 | 0.7391 |
| **Optuna Tuned CatBoost** | Bayesian TPE search on class weights + speed gate | 0.7648 | 0.4348 | 0.8889 |
| **AutoGluon (63 Feats, Default Argmax)** | 5-Fold Bagged ensemble (LGBM, XGB, CatBoost, ET) | 0.7842 | 0.4706 | 0.9140 |
| **AutoGluon (92 Feats, Default Argmax)** | 5-Fold Bagged ensemble on spatial + wavelength features | 0.8028 | 0.5263 | 0.9130 |
| **Winning Production Ensemble** | **Bagged LightGBM + LightGBMXT + Threshold Calibration** | **0.8298** | **0.5882** | **0.9333** |

---

## 5. Out-of-Fold Decision Threshold Optimization

### 5.1 The Minority Class Under-Prediction Problem
Standard multi-class decision rules select $\hat{y} = \arg\max_c P(y=c \mid X)$. Because `Side I` is only 5.1% of the training distribution, well-calibrated posterior probabilities rarely exceed 0.50, causing standard argmax to miss over 64% of real corrugations.

### 5.2 Calibrated Divisor Formulation
To maximize Macro F1 without retraining the models, class-specific divisors $\mathbf{d} = [d_{\text{Normal}}, d_{\text{Side I}}, d_{\text{Side II}}]$ are fitted on the **out-of-fold probability distributions**:
$$\hat{y}_i = \arg\max_{c \in \{0, 1, 2\}} \left( \frac{P_{i, c}}{d_c} \right)$$
Subject to: $d_{\text{Normal}} = 1.0$, $d_{\text{Side I}} \in [0.10, 0.50]$, $d_{\text{Side II}} \in [0.60, 1.20]$.

### 5.3 Optimal Solution
$$\mathbf{d}^* = [1.00,\, 0.17,\, 0.75]$$
- Effect on Side I: Lowers the effective positive threshold from $0.50$ to $\approx 0.15\text{--}0.20$.
- Effect on Side II: Slight adjustment preserving 100% precision.

### 5.4 Final Out-of-Fold Confusion Matrix (272 Files)
```text
                   Predicted Normal    Predicted Side I    Predicted Side II    Total
Actual Normal            225                  9                   0              234
Actual Side I              4                 10                   0               14
Actual Side II             2                  1                  21               24
Total                    231                 20                  21              272
```

### 5.5 Final Metric Breakdown
- **Macro F1**: **0.8298**
- **Overall Accuracy**: **94.12%**
- **Side I**: Precision = 50.00%, **Recall = 71.43% (10/14 caught)**, F1 = 0.5882 (or 0.6250 in 3-model blend)
- **Side II**: **Precision = 100.00%**, Recall = 87.50%, **F1 = 0.9333**
- **Normal**: Precision = 97.40%, Recall = 96.15%, F1 = 0.9677

---

## 6. Production Artifacts & File Structure

All production artifacts have been consolidated inside `Rail_Corrugation/`:

```text
Rail_Corrugation/
├── models/
│   ├── rail_model_bundle.joblib    # ~3MB self-contained production bundle (no PyTorch/GPU needed)
│   └── rail_pipeline.py            # Clean Python inference engine (RailPredictor class)
├── automl_rail_predictions.csv     # Official 68-test-file submission (file_id, prediction)
├── baseline_rail_predictions.csv   # Mirrored submission file
├── test_predictions_detailed.csv   # Detailed predictions with confidence and speed diagnostics
├── automl_leaderboard_results.csv  # Model comparison leaderboard table
├── rail_all_features.csv           # 272 x 92 cached training feature matrix
└── train_enhanced_model.py         # End-to-end training, CV, and export script
```

---

## 7. Inference & API Verification

The production inference pipeline in `models/rail_pipeline.py` requires only `lightgbm`, `numpy`, `pandas`, `scipy`, and `joblib`.

```python
from models.rail_pipeline import RailPredictor

# 1. Initialize engine
predictor = RailPredictor()

# 2. Predict on raw file or DataFrame
result = predictor.predict("Test1.csv")

# 3. Verified output structure:
# {
#   "prediction": "Normal",
#   "confidence": 0.9791,
#   "speed_kmh": 35.57,
#   "status": "Normal Healthy Track",
#   "probabilities": {"Normal": 0.9791, "Side I": 0.0074, "Side II": 0.0134},
#   "diagnostic_summary": "Model classified track recording as Normal at 35.57 km/h with 97.9% calibrated confidence.",
#   "asymmetry_ratio": 0.983,
#   "top1_car_ratio": 1.898,
#   "top2_car_ratio": 1.53,
#   "dominant_wavelength_s1_cm": 10.12
# }
```

### Computational Performance
- Feature extraction per 10,000 $\times$ 129 recording: **$\approx 260\text{ ms}$**
- Ensemble inference (10 bagged LightGBM trees): **$\approx 1\text{ ms}$**
- Total latency per file: **$< 300\text{ ms}$** on CPU.
