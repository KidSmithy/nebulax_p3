# Exploratory Data Analysis (EDA) Report: Rail Corrugation Sensor Dataset

> **Document Purpose**: Comprehensive technical and statistical analysis of the multi-channel sensor dataset for railway track condition monitoring. This document breaks down the recording geometry, physical mechanics, channel mappings, class distributions, kinematic speed profiles, time-frequency characteristics, and defect signal signatures.

---

## 1. Dataset Dimensions & Recording Architecture

The dataset captures dynamic railway vehicle-track interaction over fixed 1.0-second inspection segments:

| Dimension | Training Partition | Test Partition | Total Combined |
| :--- | :---: | :---: | :---: |
| **Recordings Count** | 272 files | 68 files | 340 files |
| **Duration per File** | 1.000 second | 1.000 second | 1.000 second |
| **Sampling Frequency ($f_s$)** | 10,000 Hz (10 kHz) | 10,000 Hz (10 kHz) | 10,000 Hz (10 kHz) |
| **Time Steps per File ($T$)** | 10,000 steps | 10,000 steps | 10,000 steps |
| **Channels per File ($C$)** | 129 channels | 129 channels | 129 channels |
| **Data Points per File** | 1,290,000 points | 1,290,000 points | 1,290,000 points |
| **File Format** | CSV | CSV | CSV |

---

## 2. Sensor Layout & Channel Spatial Mapping

Each recording contains exactly **129 columns**, representing an 8-car train configuration with dual-rail instrumentation:

```text
[Column 0] Rotating speed (Axle Speed Pulse Sensor)
├── [Columns 1 - 64]   Bearing Vibration Sensors (8 Cars × 8 Positions = 64 channels)
│   ├── Side I (Left/Inner Rail)  : Positions 1, 3, 5, 7 across Cars 1–8 (32 channels)
│   └── Side II (Right/Outer Rail): Positions 2, 4, 6, 8 across Cars 1–8 (32 channels)
└── [Columns 65 - 128] Bearing Shock Sensors (8 Cars × 8 Positions = 64 channels)
    ├── Side I (Left/Inner Rail)  : Positions 1, 3, 5, 7 across Cars 1–8 (32 channels)
    └── Side II (Right/Outer Rail): Positions 2, 4, 6, 8 across Cars 1–8 (32 channels)
```

### 2.1 Physical Geometry of the Train & Contact
- **Train Length**: An 8-car passenger consist is approximately **160 meters** long (~20 m per car).
- **Temporal Advancement**: At an operating speed of $50\text{ km/h} \approx 13.89\text{ m/s}$, the train advances only **13.89 meters** during the 1.0-second recording window.
- **Critical Insight**: Because $13.89\text{ m} \ll 160\text{ m}$, a localized track defect on the rail head is **not felt equally by all 8 cars**. During any given 1-second file, only 1 or 2 cars (e.g., Car 7 and Car 8) pass directly over the corrugated patch. Global averaging across all 8 cars severely dilutes the fault signal.

---

## 3. Label Distribution & Severe Class Imbalance

The training labels (`Train_Labels.csv`) classify each recording into one of three states:

```text
Class Distribution (N = 272):
┌─────────────────────────────────────────────────────────────┐
│ Normal   ██████████████████████████████████████  234 (86.03%)│
│ Side II  ████                                     24  (8.82%)│
│ Side I   ██                                       14  (5.15%)│
└─────────────────────────────────────────────────────────────┘
```

| Class | Label ID | Sample Count | Percentage | Physical Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **`Normal`** | 0 | 234 | 86.03% | Healthy track or stationary/low-speed non-corrugated segments. |
| **`Side II`** | 2 | 24 | 8.82% | Abnormal periodic corrugation on the Right/Outer rail. |
| **`Side I`** | 1 | 14 | 5.15% | Abnormal periodic corrugation on the Left/Inner rail. |

### The "Side I" Statistical Bottleneck:
- `Side I` contains **only 14 positive examples** out of 272 recordings.
- In a standard 5-fold cross-validation split, each validation fold contains **only 2 or 3 Side I samples**.
- A single false negative drops `Side I` recall by $33\%$. This extreme sparsity causes standard objective functions (cross-entropy/accuracy) to ignore Side I unless specifically counter-balanced with decision threshold calibration and class weighting.

---

## 4. Kinematics & Train Speed Analysis

The axle speed channel (`Rotating speed`) records discrete pulse transitions from a 180-tooth wheel encoder.

### 4.1 Kinematic Speed Decoding Formula
$$\Delta s_t = |s_t - s_{t-1}|$$
$$N_{\text{trans}} = \sum_{t=1}^{T-1} \mathbb{I}(\Delta s_t > 0)$$
$$\text{Revolutions} = \frac{N_{\text{trans}}}{180}$$
$$\text{Linear Velocity } v = \text{Revolutions} \times (\pi \times D_{\text{wheel}}) \quad \left(D_{\text{wheel}} = 0.85\text{ m}, \text{Circumference} = 2.67035\text{ m}\right)$$
$$v_{\text{km/h}} = v \times 3.6$$

### 4.2 Speed Distribution Across Health Classes

| Health Class | Minimum Speed | Median Speed | Mean Speed | Maximum Speed | Speed Std Dev |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`Normal`** | 0.00 km/h | 28.52 km/h | 28.88 km/h | 70.12 km/h | 21.45 km/h |
| **`Side I`** | **34.93 km/h** | 49.35 km/h | 49.45 km/h | 66.97 km/h | 9.82 km/h |
| **`Side II`** | **42.08 km/h** | 56.40 km/h | 52.51 km/h | 66.65 km/h | 7.94 km/h |

```text
Speed Ranges (km/h):
Normal : [0.0 ========================================================= 70.1]
Side I :                       [34.9 ======================== 67.0]
Side II:                            [42.1 =================== 66.7]
         |-----------|-----------|-----------|-----------|-----------|
         0          15          30          45          60          75 km/h
```

### 4.3 The Critical Low-Speed Physical Gate ($< 10.0\text{ km/h}$)
- **Observation**: In the training partition, **100% of samples with $v < 10.0\text{ km/h}$ are `Normal`**. Not a single corrugation defect appears below 34.9 km/h.
- **Physical Explanation**: Dynamic rail corrugation excitation requires rolling velocity to induce wheel-rail dynamic contact resonance ($20\text{--}1000\text{ Hz}$). At creeping speeds or when stopped ($0\text{ km/h}$), sensor readings reflect only ambient noise and electrical drift.
- **Test Set Distribution**: In the unlabelled 68 test recordings, exactly **16 files (23.5%)** are stationary or creeping below 10 km/h:
  - 8 files are completely stationary ($v = 0.00\text{ km/h}$).
  - 8 files are crawling between $3.15\text{ km/h}$ and $8.87\text{ km/h}$.
  - Gating these files deterministically to `Normal` prevents low-speed sensor drift from producing false alarms.

---

## 5. Signal Properties & Time-Frequency Analysis

### 5.1 DC Offset & Static Gravitational Bias
- Raw sensor channels exhibit static DC offsets up to $\pm 0.5\text{ m/s}^2$ caused by sensor mounting angle and gravitational acceleration.
- **Pre-processing Requirement**: Channel-wise mean centering ($\tilde{x}(t) = x(t) - \mu_x$) is required before computing energy or spectral metrics to prevent DC energy from overwhelming the dynamic frequency bands.

### 5.2 Vibration vs Shock Mechanics
1. **Vibration Sensors (0–4000 Hz)**:
   - Capture continuous rolling contact and steady harmonic resonance.
   - Primary metric: **RMS Energy** ($\sqrt{\frac{1}{T}\sum \tilde{x}^2}$) and **Left-to-Right RMS Ratio** ($V_{\text{Side I}} / V_{\text{Side II}}$).
2. **Shock Sensors (Impulse / Impact)**:
   - Capture transient metal-on-metal impacts from wheel pits, rail joints, or severe short-pitch corrugation.
   - Primary metric: **Kurtosis** ($\frac{\mu_4}{\sigma^4}$) and **Crest Factor** ($\frac{\text{Peak}}{\text{RMS}}$). Normal track exhibits near-Gaussian kurtosis ($\approx 3\text{--}6$), whereas shock channels on severe defects reach kurtosis $> 15$.

### 5.3 Spectral Signatures & Spatial Wavelength Bands
Rail corrugations have fixed physical wavelengths on the steel surface ($\lambda$), but the excitation frequency observed by the train sensor shifts directly with velocity:
$$f = \frac{v}{\lambda} \iff \lambda = \frac{v}{f}$$

For typical railway corrugation regimes ($2\text{ cm} \le \lambda \le 30\text{ cm}$) at operational speeds ($35\text{--}70\text{ km/h} \approx 9.7\text{--}19.4\text{ m/s}$):
- $\lambda = 30\text{ cm} \implies f \approx 32\text{--}65\text{ Hz}$
- $\lambda = 10\text{ cm} \implies f \approx 97\text{--}194\text{ Hz}$
- $\lambda = 4\text{ cm} \implies f \approx 243\text{--}486\text{ Hz}$
- $\lambda = 2\text{ cm} \implies f \approx 486\text{--}972\text{ Hz}$

Spectral analysis reveals that **corrugations manifest as pronounced energy concentration in the $100\text{--}1000\text{ Hz}$ band** specifically on the defective rail side, with prominent spatial wavelengths in the **$4\text{--}12\text{ cm}$ range**.

---

## 6. Detailed Defect Signatures: Side I vs. Side II vs. Normal

### 6.1 Statistical Metric Comparison (Operational Speeds $v \ge 30\text{ km/h}$)

| Metric | Normal (Healthy) | Side I Corrugation | Side II Corrugation |
| :--- | :---: | :---: | :---: |
| **Vibration RMS Ratio ($S_1 / S_2$)** | $0.992 \pm 0.08$ (Balanced) | **$1.218 \pm 0.28$ (Left Heavy)** | **$0.485 \pm 0.12$ (Right Heavy)** |
| **Vibration RMS Diff ($S_1 - S_2$)** | $-0.010\text{ m/s}^2$ | **$+0.080\text{ m/s}^2$** | **$-0.220\text{ m/s}^2$** |
| **Speed-Normalized RMS Diff** | $-0.014$ | **$+0.124$** | **$-0.315$** |
| **Primary Peak Car Ratio** | $1.15 \pm 0.12$ | **$1.68 \pm 0.52$** | **$0.42 \pm 0.08$** |
| **Tail Cars Ratio (Cars 6–8)** | $0.98 \pm 0.09$ | **$1.38 \pm 0.41$** | **$0.51 \pm 0.11$** |
| **Wave Band 7–12 cm Ratio** | $1.01 \pm 0.10$ | **$1.42 \pm 0.35$** | **$0.58 \pm 0.14$** |
| **Band 250–500 Hz Side I Energy** | $6.7 \times 10^{-5}$ | **$9.3 \times 10^{-5}$** | $4.2 \times 10^{-5}$ |

### 6.2 Why Side II is Easier to Detect than Side I
- **Side II Defect Severity**: The 24 Side II recordings have a massive average asymmetry ($S_1 / S_2 \approx 0.48$, diff $\approx -0.22\text{ m/s}^2$). The defect is deep and high-amplitude. Classifiers achieve $>92\%$ precision and $>84\%$ recall effortlessly.
- **Side I Defect Subtlety**: The 14 Side I recordings have a milder average asymmetry ($S_1 / S_2 \approx 1.22$, diff $\approx +0.08\text{ m/s}^2$). In two files (`Train185.csv` and `Train202.csv`), the maximum car ratio barely reaches $1.07\text{--}1.15$. 
- **Car Localization**: In 10 of the 14 Side I files, the asymmetry spikes almost exclusively in **Car 7 and Car 8**. If features only look at whole-train averages, Side I gets swallowed by the 6 healthy cars.

---

## 7. Key Data Takeaways & Feature Engineering Recommendations

1. **Do Not Rely on Raw Temporal Matrices**: The 10,000 $\times$ 129 structure contains phase shifts from wave propagation along the bogies. Frequency domain (Welch PSD) and spatial domain summary metrics are far more robust.
2. **Remove the Speed Confounder**: Vibration scales with $v^2$. Always normalize vibration energy by $(v / 50)^2$ to avoid confusing high-speed normal track with moderate-speed corrugated track.
3. **Use Car Order Statistics**: Track the top 1, top 2, and trailing car ratios (`top1_car_ratio`, `mean_top2_car_ratio`, `tail_cars_v_ratio`) to catch localized defects that only affect 1–2 wheelsets.
4. **Enforce the $< 10\text{ km/h}$ Gate**: Stationarity eliminates corrugation excitation. Low-speed gating avoids unnecessary false positives.
5. **Calibrate Decision Thresholds**: Standard 50% probability cutoffs fail on the 5.1% Side I class. Post-hoc or nested divisor calibration ($\approx 0.16\text{--}0.20$ effective threshold) is required to achieve high minority recall.
