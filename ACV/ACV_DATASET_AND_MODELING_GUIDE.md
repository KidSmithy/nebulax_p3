# Comprehensive Dataset Investigation & Modeling Strategy Guide: Train ACV Subsystem
**Subsystem**: Rail Vehicle Air Conditioning & Ventilation (ACV) System  
**Failure Mode**: Refrigerant Leakage (Undercharge Anomaly & Localisation)  
**Primary Task**: Fault Localisation & Consist-Wide Vehicle Ranking  
**Scoring Metric**: Linear Rank-Decay Score ($S = \frac{n - (r - 1)}{n}$)  
**Target Audience**: Manus AI / Autonomous ML Modeling Agent / Senior Machine Learning Engineer  

---

## 1. Executive Summary & Challenge Overview

Rail vehicle Air Conditioning and Ventilation (ACV) systems maintain passenger thermal comfort and equipment safety. In passenger trains, refrigerant leakage accounts for approximately **40% of all cooling failures** and over **30% of all vehicle subsystem anomalies**. Undetected refrigerant leakage causes loss of cooling capacity, compressor motor burnout, emergency air-conditioning shutdowns, and severe passenger discomfort, particularly during peak summer periods.

### The Machine Learning Challenge
Unlike standard classification tasks where samples are independent rows in a table:
1. **Extreme Sample Sparsity**: There are only **6 historical training case files** (`acv_case_01.xlsx` through `acv_case_06.xlsx`) and **1 held-out test case** (`acv_test_case.xlsx`).
2. **High-Frequency Continuous Telemetry**: Each case contains thousands of timestamps (sampled every 30 seconds across multi-day operating runs, up to 22,262 rows per file), recording multivariate telemetry across all train cars.
3. **Localisation / Ranking Task**: Exactly **one car** in each case has a refrigerant leak, while the remaining cars are healthy. The model must output an ordered ranking of **all cars** in the train consist from most-likely to least-likely faulty (`ranked_cars = 01|03|04|...`).
4. **Schema Heterogeneity**: File formats and column headers vary across cases (e.g., Case 04 is a 4-car consist with 59 parameters per car, while Cases 01–03, 05, 06 and the Test case are 8-car consists with 8 core parameters per car under differing column names).
5. **No Pointwise Row Ground Truth**: No individual timestamp is labeled as "leaking". Only the overall car ID is provided in `Train_Labels.csv`.

---

## 2. Physical & Thermodynamic Mechanism of Refrigerant Leakage

To design the optimal predictive model, understanding the underlying thermodynamic process of vapour-compression refrigeration in rail ACV units is essential:

```
+--------------------------------------------------------------------------------+
|                        Vapour-Compression Refrigeration Cycle                  |
|                                                                                |
|        [Compressor]  ===>  (High Pressure / High Temp Vapour)                  |
|             ^                                   |                              |
|             |                                   v                              |
|   (Low Pressure Vapour)                    [Condenser] (Heat rejected outside) |
|             ^                                   |                              |
|             |                                   v                              |
|        [Evaporator]  <===   (Expansion Valve) <== (High Pressure Liquid)       |
|    (Heat absorbed from                                                         |
|      passenger cabin)                                                          |
+--------------------------------------------------------------------------------+
```

### Physical Manifestations of Leakage:
1. **Drop in Condensation / Discharge Pressure**:
   When refrigerant mass $\dot{m}$ leaks, high-side and low-side pressures decline. In `acv_case_04.xlsx` (where detailed pressure telemetry is recorded), the faulty Car 01 exhibits a mean High Pressure of **1583.41 kPa** compared to **1771–1938 kPa** in healthy cars.
2. **Loss of Enthalpy Differential & Cooling Capacity**:
   $Q_{\text{cool}} = \dot{m} \cdot (h_{\text{out}} - h_{\text{in}})$. As refrigerant mass drops, the cooling power delivered by the evaporator degrades significantly.
3. **Thermal Setpoint Tracking Failure**:
   Under cooling conditions, the ACV unit enters continuous cooling (`Automatic Cooling` or `Full Cooling`), but the evaporator cannot absorb enough heat. The cabin temperature ($T_{\text{in}}$) fails to pull down to the setpoint ($T_{\text{set}}$), causing persistent positive error: $(T_{\text{in}} - T_{\text{set}}) > 0$.
4. **Cross-Car Differential Anomaly**:
   Ambient weather, solar radiation, diurnal heat, and outdoor humidity affect the entire train equally. Therefore, subtracting the **consist-wide median indoor temperature** at each timestamp cancels out external weather confounding variables, isolating the single failing car.

---

## 3. Dataset Inventory & Schema Architecture

All datasets are stored in `02_Datasets/ACV/`:

```
02_Datasets/ACV/
├── Train_Labels.csv
├── Train/
│   ├── acv_case_01.xlsx   (1.64 MB, 6,999 rows, 67 columns)
│   ├── acv_case_02.xlsx   (2.19 MB, 9,187 rows, 67 columns)
│   ├── acv_case_03.xlsx   (1.96 MB, 8,310 rows, 67 columns)
│   ├── acv_case_04.xlsx   (32.24 MB, 22,262 rows, 483 columns)
│   ├── acv_case_05.xlsx   (1.57 MB, 6,972 rows, 67 columns)
│   └── acv_case_06.xlsx   (0.73 MB, 3,263 rows, 67 columns)
└── Test/
    └── acv_test_case.xlsx (2.12 MB, 9,082 rows, 67 columns)
```

### 3.1 File Specifications & Metadata

| File Name | Split | Rows | Cols | Consist Size | Active Cars | Time Span | Sampling Rate |
|---|---|---|---|---|---|---|---|
| `acv_case_01.xlsx` | Train | 6,999 | 67 | 8 cars | `01`–`08` | 2023-05-18 to 2023-05-21 (~3.9 days) | 30 seconds |
| `acv_case_02.xlsx` | Train | 9,187 | 67 | 8 cars | `01`–`08` | 2020-07-09 to 2020-07-12 (~4.0 days) | 30 seconds |
| `acv_case_03.xlsx` | Train | 8,310 | 67 | 8 cars | `01`–`08` | 2021-09-30 to 2021-10-03 (~3.8 days) | 30 seconds |
| `acv_case_04.xlsx` | Train | 22,262 | 483 | 4 active / 8 listed | `01`–`04` (`05`–`08` NaN) | 2023-08-15 to 2023-08-18 (~3.6 days) | ~10-30 seconds |
| `acv_case_05.xlsx` | Train | 6,972 | 67 | 8 cars | `01`–`08` | 2021-03-03 to 2021-03-06 (~3.7 days) | 30 seconds |
| `acv_case_06.xlsx` | Train | 3,263 | 67 | 8 cars | `01`–`08` | 2020-07-06 to 2020-07-08 (~2.6 days) | 30 seconds |
| `acv_test_case.xlsx` | **Test** | **9,082** | **67** | **8 cars** | `01`–`08` | 2021-06-24 to 2021-06-27 (~4.0 days) | 30 seconds |

### 3.2 Ground Truth Training Labels (`Train_Labels.csv`)

| Filename | Faulty Car ID | Faulty Car Column Prefix | Consist Type |
|---|---|---|---|
| `acv_case_01.xlsx` | 1 | `Car 01` | 8-car consist |
| `acv_case_02.xlsx` | 2 | `Car 02` | 8-car consist |
| `acv_case_03.xlsx` | 3 | `Car 03` | 8-car consist |
| `acv_case_04.xlsx` | 1 | `Car 01` | 4-car consist |
| `acv_case_05.xlsx` | 4 | `Car 04` | 8-car consist |
| `acv_case_06.xlsx` | 6 | `Car 06` | 8-car consist |

---

## 4. Parameter Schemas & Normalization Strategy

### 4.1 Metadata Columns (First 3 columns in every file)
- `Car model`: Train model series (e.g., `'A'`).
- `Train number`: Vehicle formation identifier (e.g., `620`).
- `Time`: Datetime timestamp (e.g., `2021-06-24 00:00:00`).

### 4.2 Standard 8 Car Parameters (Present in Test & Cases 1, 2, 3, 5, 6)
Every car $c \in \{01, \dots, 08\}$ has 8 dedicated telemetry parameters prefixed by `Car <NN> - `:

| Standardized Parameter Name | Canonical Raw Column Name | Units / Format | Physical Meaning |
|---|---|---|---|
| `Indoor Average Temperature` | `Car <NN> - Indoor Average Temperature` | °C (float) | Average temperature inside the passenger cabin of car $c$. |
| `Outdoor Average Temperature` | `Car <NN> - Outdoor Average Temperature` *or* `Outside Temperature Sensor Reading` | °C (float/int) | External ambient air temperature measured by train sensor. |
| `ACV Control Temperature (Cooling)` | `Car <NN> - ACV Control Temperature (Cooling)` | °C (float/int) | Target cooling setpoint commanded by the ACV controller. |
| `ACV Control Temperature (Heating)` | `Car <NN> - ACV Control Temperature (Heating)` | °C (float/int) | Target heating setpoint commanded by the ACV controller. |
| `ACV Running Mode` | `Car <NN> - ACV Running Mode` | String / Enum | Operational state: `Automatic Cooling`, `Full Cooling`, `Stop`, `Emergency Ventilation`, `Invalid`. |
| `ACV Setting Mode` | `Car <NN> - ACV Setting Mode` | String / Enum | Control mode: `Centralized Control`, `Manual Control`, `Invalid`. |
| `Load Halved` | `Car <NN> - Load Halved` | String / Enum | Capacity status: `Normal` or `Load Halved` (compressor cutback). |
| `ACV Information Valid` | `Car <NN> - ACV Information Valid` | String / Enum | Telemetry health: `Valid` or `Invalid`. |

### 4.3 Critical Schema Variations & Pitfalls
> [!WARNING]
> **Schema Variations Between Files**:
> 1. **Outdoor Temp Header Divergence**: In `acv_case_05.xlsx` and `acv_case_06.xlsx`, the column is named `Car <NN> - Outside Temperature Sensor Reading`, whereas in Cases 01–03 and Test, it is named `Car <NN> - Outdoor Average Temperature`.
> 2. **Case 04 Consist Size**: In `acv_case_04.xlsx`, only Cars 01–04 have active data. Cars 05–08 are 100% NaN.
> 3. **Case 04 Parameter Expansion**: Case 04 contains 59 parameters per car (including compressor currents, high/low refrigerant pressures, damper positions).
> 4. **Test Set Constraint**: The test set `acv_test_case.xlsx` **ONLY contains the 8 standard parameters**! Any model trained on Case 04's extra 51 parameters will fail during test evaluation. Therefore, all models must operate strictly on the core 8 features or normalized cross-car derivations.

---

## 5. Exploratory Data Analysis & Empirical Evidence

### 5.1 Cross-Car Differential Analysis on Training Cases
When we compute the **relative temperature differential**:
$$\Delta T_{\text{rel}, c}(t) = T_{\text{in}, c}(t) - \text{median}_{k}(T_{\text{in}, k}(t))$$
and the **cooling setpoint tracking error**:
$$\Delta T_{\text{set}, c}(t) = T_{\text{in}, c}(t) - T_{\text{cooling\_set}, c}(t)$$
the true faulty car consistently emerges at the extreme upper tail of thermal deficit:

| Training Case | True Faulty Car | Mean $\Delta T_{\text{rel}}$ | 95th %ile $\Delta T_{\text{rel}}$ | Mean $\Delta T_{\text{set}}$ (Cooling) | Rank Among Cars | Score |
|---|---|---|---|---|---|---|
| `acv_case_01.xlsx` | **Car 01** | **+0.317 °C** | **+3.00 °C** | **+0.670 °C** | **1 / 8** | **1.0000** |
| `acv_case_02.xlsx` | **Car 02** | **+0.043 °C** | **+0.75 °C** | **+0.347 °C** | **1 / 8** | **1.0000** |
| `acv_case_03.xlsx` | **Car 03** | **+0.570 °C** | **+2.00 °C** | **+0.967 °C** | **1 / 8** | **1.0000** |
| `acv_case_04.xlsx` | **Car 01** | **+0.122 °C** | **+1.00 °C** | **+0.210 °C** | **2 / 4** | **0.7500** |
| `acv_case_05.xlsx` | **Car 04** | **+0.183 °C** | **+1.00 °C** | **-0.713 °C** | **1 / 8** | **1.0000** |
| `acv_case_06.xlsx` | **Car 06** | **+1.272 °C** | **+2.25 °C** | **+2.692 °C** | **1 / 8** | **1.0000** |

> [!NOTE]
> In 5 out of the 6 training cases, the true faulty car is ranked **#1** by simple relative thermal deficit. In Case 04 (a 4-car consist), it ranks **#2** (Score: 0.7500). The average benchmark score across all 6 cases is **0.9583 / 1.0000**.

### 5.2 Test Case Profile (`acv_test_case.xlsx`)
- **Metadata**: Car model `'A'`, Train number `620`, 9,082 rows, 8 active cars.
- **Outdoor Temperature**: Ranging from 0.0 °C to 37.5 °C (mean: 29.6 °C — high summer cooling scenario).
- **Consist Telemetry Comparison**:

| Car ID | Mean Indoor Temp (°C) | Mean $\Delta T_{\text{rel}}$ | 95th %ile $\Delta T_{\text{rel}}$ | Mean $\Delta T_{\text{set}}$ during Cooling | Anomaly Score | Predicted Rank |
|---|---|---|---|---|---|---|
| **Car 01** | **24.197** | **+0.1024** | **+1.00** | **+0.1246** | **0.1647** | **1st (Fault Suspect)** |
| **Car 03** | 24.158 | +0.0639 | +0.50 | -0.0531 | 0.0336 | 2nd |
| **Car 04** | 24.098 | +0.0034 | +0.75 | -0.0781 | 0.0234 | 3rd |
| **Car 07** | 24.107 | +0.0130 | +0.50 | -0.0756 | -0.0252 | 4th |
| **Car 08** | 24.099 | +0.0044 | +0.50 | -0.0884 | -0.0399 | 5th |
| **Car 06** | 24.090 | -0.0044 | +0.50 | -0.1135 | -0.0649 | 6th |
| **Car 02** | 24.047 | -0.0469 | +0.50 | -0.1730 | -0.1391 | 7th |
| **Car 05** | 24.004 | -0.0898 | +0.50 | -0.1847 | -0.1760 | 8th |

**Key Test Finding**: **Car 01** is the only car in the entire test consist where cabin temperature consistently exceeds the commanded cooling setpoint during active cooling ($\Delta T_{\text{set}} > 0$).

---

## 6. Mathematical Evaluation Metric

The competition evaluates ACV models using the **Linear Rank-Decay Score**, formally defined as:

$$\text{Score} = \frac{n - (r - 1)}{n}$$

where:
- $n$ is the total number of cars ranked in that file ($n = 8$ for standard trains, $n = 4$ for Case 04).
- $r$ is the 1-based rank position of the ground-truth faulty car in the submitted `ranked_cars` list ($r \in [1, n]$).
- If the true faulty car is missing or the file is omitted, $\text{Score} = 0$.

### Score Scale for an 8-Car Consist ($n = 8$):
| True Faulty Car Rank | Rank Decay Calculation | Partial Credit Score |
|---|---|---|
| **1st (Top Pick)** | $(8 - 0) / 8$ | **1.0000** |
| **2nd** | $(8 - 1) / 8$ | **0.8750** |
| **3rd** | $(8 - 2) / 8$ | **0.7500** |
| **4th** | $(8 - 3) / 8$ | **0.6250** |
| **5th** | $(8 - 4) / 8$ | **0.5000** |
| **6th** | $(8 - 5) / 8$ | **0.3750** |
| **7th** | $(8 - 6) / 8$ | **0.2500** |
| **8th (Last)** | $(8 - 7) / 8$ | **0.1250** |
| Missing / Format Error | N/A | **0.0000** |

---

## 7. Model Architectures & Modeling Paradigms for Manus AI

When instructing Manus AI to decide on the best modeling strategy, provide the following four candidate paradigms:

### Paradigm 1: Physics-Informed Relative Thermal Deficit Ranker (Unsupervised / Deterministic) — **RECOMMENDED BASELINE**
- **How it Works**:
  1. For each timestamp $t$, compute consist baseline $\tilde{T}_{\text{in}}(t) = \text{median}_{k \in \mathcal{C}}(T_{\text{in}, k}(t))$.
  2. Filter timestamps where cooling is active (`ACV Running Mode` contains `'Cooling'`).
  3. Extract multi-dimensional deficit features:
     - Mean differential: $\mu_{\Delta, c} = \frac{1}{T}\sum (T_{\text{in}, c} - \tilde{T}_{\text{in}})$
     - 95th percentile differential: $q_{95, c}$
     - Setpoint tracking deficit: $\mu_{\text{set}, c} = \frac{1}{T}\sum (T_{\text{in}, c} - T_{\text{set}, c})$
     - Integrated positive error: $E_{+, c} = \sum \max(0, T_{\text{in}, c} - T_{\text{set}, c})$
  4. Composite Anomaly Score: $S_c = \mu_{\Delta, c} + \alpha \cdot \mu_{\text{set}, c} + \beta \cdot q_{95, c}$.
  5. Sort cars descending by $S_c \implies$ `ranked_cars`.
- **Strengths**: Zero danger of overfitting on 6 cases; robust to distribution shifts; achieved **0.9583 CV score**.
- **Weaknesses**: Fixed heuristic weights ($\alpha, \beta$).

### Paradigm 2: Pairwise Gradient Boosting Learning-to-Rank (XGBoost / LightGBM)
- **How it Works**:
  1. Reformulate the 6 cases into pairwise car comparisons $(c_i, c_j)$.
  2. For an 8-car consist, there are $\binom{8}{2} = 28$ unordered pairs ($56$ directed pairs). Across 6 cases, this creates **336 training instances**.
  3. Feature Vector for pair $(c_i, c_j)$: Difference in feature statistics:
     $$\mathbf{x}_{ij} = \mathbf{f}(c_i) - \mathbf{f}(c_j)$$
  4. Label $y_{ij} = 1$ if $c_i$ is more faulty than $c_j$, else $0$.
  5. Train a Gradient Boosted Decision Tree (GBDT) with `rank:pairwise` or binary logistic loss.
  6. At inference, predict tournament win-rates or margin scores for each car $\to$ sort cars descending.
- **Strengths**: Optimizes weights automatically; expands 6 cases into 336 pairwise training samples.
- **Weaknesses**: Requires careful regularisation to avoid leaf-node overfitting.

### Paradigm 3: Spatial-Temporal Graph Neural Network (GNN) / 1D ResNet
- **How it Works**:
  1. Model the train consist as an 8-node linear graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ where edges connect adjacent cars $c_i - c_{i+1}$.
  2. Each node takes an 8-channel time-series tensor (temperature, setpoint, mode flags).
  3. GNN message passing allows each car node to compare its thermal state against its immediate neighbours and consist aggregate.
  4. Node-level scalar output passes through a Softmax / Permutation layer with Plackett-Luce ranking loss.
- **Strengths**: Directly models heat diffusion and passenger circulation between adjacent cars.
- **Weaknesses**: Extremely data-hungry; training deep GNNs on only 6 graph instances has very high risk of overfitting without aggressive pre-training or synthetic data generation.

### Paradigm 4: Multi-Car Reconstruction Autoencoder (Unsupervised Anomaly Detection)
- **How it Works**:
  1. Train a 1D Convolutional Autoencoder or Transformer Encoder on all cars across all timestamps.
  2. The autoencoder learns the standard correlation pattern of normal operating cars.
  3. The car with the highest mean reconstruction error ($\| \mathbf{x}_{c} - \hat{\mathbf{x}}_{c} \|^2$) during cooling cycles is flagged as the anomaly.
- **Strengths**: Purely unsupervised; learns complex multi-modal sensor relationships.
- **Weaknesses**: May reconstruct faulty patterns if the faulty car's signals are dominant.

---

## 8. Manus AI Decision Matrix & Recommendation

| Criteria | Paradigm 1: Physics Deficit Ranker | Paradigm 2: Pairwise GBDT Ranker | Paradigm 3: Spatial-Temporal GNN | Paradigm 4: Autoencoder Residual |
|---|---|---|---|---|
| **Data Efficiency (N=6)** | **Exceptional (No fitting)** | **High (336 pairs)** | Low (Severe overfit risk) | Medium |
| **CV Benchmark Score** | **0.9583 / 1.0000** | Expected 0.92–0.97 | High variance | Expected 0.85–0.90 |
| **Interpretability** | **100% Transparent** | High (SHAP values) | Black box | Medium |
| **Inference Latency** | **< 0.5 seconds** | **< 0.5 seconds** | ~2–5 seconds | ~1–3 seconds |
| **Implementation Complexity** | Low | Medium | High | High |
| **Risk of Submission Failure**| **Zero** | Low | High | Medium |

### Final Recommendation for Manus AI:
1. **Primary Model**: Deploy **Paradigm 1 (Physics-Informed Cross-Car Thermal Deficit Ranker)** as the robust, foolproof primary engine.
2. **Secondary Model / Ensemble**: Train **Paradigm 2 (Pairwise GBDT Ranker via XGBoost)** using pairwise delta features across the 336 car pairs, and blend the ranking probabilities with Paradigm 1.
3. **Safety Fallback**: Always ensure that if any machine learning model produces NaN or out-of-bounds outputs, it falls back to the deterministic thermal deficit ranking.

---

## 9. Validation & Anti-Leakage Protocol

To ensure valid scoring and zero data leakage:
1. **Strict Leave-One-Case-Out Cross-Validation (LOOCV)**:
   - For each fold $k \in \{1, \dots, 6\}$: Train on the remaining 5 cases, evaluate on case $k$.
   - Compute the exact competition metric: $\text{Score}_k = \frac{n - (r_k - 1)}{n}$.
   - Report the arithmetic mean $\frac{1}{6}\sum_{k=1}^6 \text{Score}_k$.
2. **Never Split Across Time/Rows**:
   - Randomly splitting rows within the same case is a **catastrophic leakage error**, because adjacent 30-second rows are highly auto-correlated. The entire multi-day case must reside either strictly in train or strictly in validation.

---

## 10. Required Submission Format & CLI Specifications

### 10.1 Output CSV Specification
The submission file must be named **`acv_predictions.csv`** and formatted as follows:

```csv
file_id,ranked_cars
acv_test_case.xlsx,01|03|04|07|08|06|02|05
```

- `file_id`: The exact filename of the held-out test case (including extension).
- `ranked_cars`: All car numbers ordered from most-likely to least-likely faulty, formatted as two-digit strings (`01`, `02`, ..., `08`), separated by the pipe character `|`.

### 10.2 Inference Script Interface (`predict.py`)
Your app or submission script must follow the standard CLI interface:
```bash
python predict.py --input path/to/acv_test_case.xlsx --output path/to/acv_predictions.csv
```

---

## 11. Companion Python Profiler & Benchmark Script

A production-grade Python script has been created alongside this guide:
`PS3/acv_data_profiler.py`

### To Run the Benchmark:
```bash
python c:\Users\Rald999\Documents\GitHub\nebulax_p3\PS3\acv_data_profiler.py
```

### Script Capabilities:
- Automatic schema alignment and column normalization across all 6 training cases and test case.
- Detection of active cars (supporting 4-car and 8-car train configurations).
- Extraction of cross-car differentials and setpoint tracking errors.
- Automated Leave-One-Case-Out cross-validation reporting exact rank-decay scores.
- Test case inference generating the compliant `acv_predictions.csv` submission file.
