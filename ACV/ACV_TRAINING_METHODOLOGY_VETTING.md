# ACV Refrigerant Leakage Fault Diagnosis: Step-by-Step Training & Methodology Dossier

**Target Audience / Purpose**: Comprehensive technical dossier for external AI model review, methodology vetting, and engineering validation.  
**Subsystem**: Train Air Conditioning & Ventilation (ACV)  
**Task**: Consist-Level Refrigerant Leakage Detection & Faulty Car Localization  
**Code Repository Location**: `ACV/` and `PS3/models/`  
**Primary Artifacts**: [`acv_pipeline.py`](file:///c:/Users/Rald999/Documents/GitHub/nebulax_p3/ACV/acv_pipeline.py), [`export_acv_model_bundle.py`](file:///c:/Users/Rald999/Documents/GitHub/nebulax_p3/ACV/export_acv_model_bundle.py), [`acv_model_bundle.joblib`](file:///c:/Users/Rald999/Documents/GitHub/nebulax_p3/ACV/acv_model_bundle.joblib)

---

## 1. Problem Formulation & Constraints

### 1.1 Objective & Output Format
- A train consist operates with $N$ passenger cars (typically $N=8$, with 4-car variants such as Case 04).
- Exactly **one** car per train journey exhibits an early-to-moderate refrigerant leak.
- The objective is not binary classification, but **complete vehicle ranking** in descending order of fault probability:
  $$\text{ranked\_cars} = c_{(1)} \mid c_{(2)} \mid \dots \mid c_{(N)} \quad \text{(e.g., '01|04|03|08|07|06|02|05')}$$
- The evaluation metric is the **Linear Rank-Decay Score**:
  $$\text{Score} = \frac{N - (r - 1)}{N}$$
  where $r \in \{1, \dots, N\}$ is the 1-based rank assigned to the true faulty car. A top-1 prediction ($r=1$) yields 1.0000; rank-2 yields 0.8750 (for $N=8$).

### 1.2 Data Realities & The "Small-$N$, High-Correlation" Challenge
- **Training Cases**: Only **6 labeled journeys** exist (`acv_case_01.xlsx` through `acv_case_06.xlsx`), each containing several thousand 1-second/10-second telemetry timestamps.
- **Test Case**: Exactly **1 held-out unlabelled journey** (`acv_test_case.xlsx`).
- **Critical Sensor Constraint**: While Case 04 includes advanced thermodynamic telemetry (compressor discharge pressure, suction pressure, motor current), **the test case (`acv_test_case.xlsx`) and the remaining training cases contain only 8 standard SCADA parameters**:
  1. `Indoor temperature` ($^\circ\text{C}$)
  2. `Outdoor temperature` ($^\circ\text{C}$)
  3. `Cooling Temperature Setpoint` ($^\circ\text{C}$)
  4. `Heating Temperature Setpoint` ($^\circ\text{C}$)
  5. `ACV Running Mode` (Cooling, Heating, Ventilation, Stop)
  6. `ACV Setting Mode`
  7. `ACV Load Halved`
  8. `ACV Information Valid`
- **Machine Learning Pitfall Avoided**: Deep neural networks (LSTM, CNN, Transformers) or row-level tabular classifiers (XGBoost predicting on raw timestamps) **must not be used**. Treating 20,000 correlated timestamps from 6 journeys as independent samples creates catastrophic data leakage, learns journey-specific ambient temperature profiles, and severely overfits.

---

## 2. Step-by-Step Training & Implementation Pipeline

The complete pipeline follows a rigorous 5-step engineering architecture:

```
[Raw Excel / CSV Telemetry]
           │
           ▼
[Step 1: Domain-Specific Cleaning & Regime Masking]
  - Normalizes heterogeneous SCADA column naming
  - Masks 'Invalid' flags to NaN before interpolation
  - Truncates CAN-bus digital dropouts (0.0°C and bounds <5°C / >45°C)
  - Prunes overnight depot shutdown rows (all-car nulls)
           │
           ▼
[Step 2: Physics-Informed Relative Thermal Deficit Extraction]
  - Filters strictly for active cooling regime ('Cooling')
  - Computes cross-car dynamic baseline: Median(T_indoor,k)
  - Calculates ΔT_rel, ΔT_set, P95 tail deficit, and temporal persistence
           │
           ▼
[Step 3: Dual-Engine Scoring Architecture]
  ┌─────────────────────────────────┴─────────────────────────────────┐
  ▼                                                                   ▼
[Engine A: Physics Health Index]               [Engine B: Pairwise Logistic Ranker]
  - Z-score normalized across cars              - Formulates all car pairs (c_i - c_j)
  - Weights: ΔT_rel (1.0), ΔT_set (0.5),        - L2-regularized logistic regression (C=0.2)
    P95 (0.25), frac>0.2°C (0.25), persist (0.25)- Outputs pairwise tournament win probabilities
  └─────────────────────────────────┬─────────────────────────────────┘
                                    │
                                    ▼
[Step 4: Hybrid Ensemble Calibration]
  - Score_final = 0.75 * Z(Physics) + 0.25 * Z(Pairwise)
  - Sorts cars in descending order of Score_final
           │
           ▼
[Step 5: Bundle Serialization & Production Inference]
  - Exports calibrated weights + model to acv_model_bundle.joblib
  - Exposes 1-line ACVPredictor interface for frontend UI / API
```

---

### Step 1: Data Cleaning & Telemetry Sanitization

Inspection revealed four real-world telemetry defects that distort statistical baselines if ignored:

1. **Sensor Dropouts ($0.0^\circ\text{C}$ CAN-bus packet errors)**:
   - In `acv_test_case.xlsx`, Car 04 contains nineteen $0.0^\circ\text{C}$ zero-readings during active cooling.
   - **Fix**: Any temperature reading $T \le 5.0^\circ\text{C}$ or $T \ge 45.0^\circ\text{C}$ is masked to `NaN`.
2. **Telemetry Validity Flag Protocol**:
   - `ACV Information Valid` indicates sensor health (`Valid` vs `Invalid`).
   - **Fix**: Data marked `Invalid` is masked to `NaN` **before** forward/backward interpolation is applied.
3. **Depot Shutdown / Power-Off Pruning**:
   - Overnight maintenance rows where all cars simultaneously report null or unpowered states are pruned.
4. **Header Normalization**:
   - Telemetry schemas vary across legacy rolling stock (e.g., `Passenger Cabin Temperature Detected Value` vs `Indoor Average Temperature`). A dictionary mapper aligns all schemas.

---

### Step 2: Feature Engineering Grounded in Refrigeration Thermodynamics

A leaking AC unit loses refrigerant charge ($R410A / R407C$), reducing evaporator cooling capacity. The compressor runs continuously, but the cabin air cannot reach the setpoint. Meanwhile, healthy cars on the same train achieve the setpoint easily.

Because all cars share identical outdoor weather, train speed, and track exposure, **the median temperature of all cars serves as an environmental baseline**:

$$\Delta T_{\text{rel}, c}(t) = T_{\text{indoor}, c}(t) - \operatorname{median}_{k \in \mathcal{C}_{\text{active}}}(T_{\text{indoor}, k}(t))$$
$$\Delta T_{\text{set}, c}(t) = T_{\text{indoor}, c}(t) - T_{\text{cooling\_setpoint}, c}(t)$$

#### Engineered Feature Set (Computed solely during active cooling):
1. **`mean_rel_diff_cooling`**: Mean elevation above the consist median during cooling ($^\circ\text{C}$).
2. **`mean_t_minus_set_cooling`**: Mean deficit between cabin temperature and cooling setpoint ($^\circ\text{C}$).
3. **`p95_rel_diff`**: 95th-percentile peak thermal delta, capturing severe load excursions under peak passenger loading.
4. **`frac_above_median`**: Proportion of active cooling duration where the car is elevated by $>0.20^\circ\text{C}$ above the consist median.
5. **`persistence_15m`**: Temporal continuity metric—fraction of time a 15-minute rolling average temperature remains elevated by $>0.20^\circ\text{C}$.

All features are converted to **within-consist Z-scores**:
$$z_{f, c} = \frac{f_c - \mu_f}{\sigma_f}$$
This ensures zero scale drift between summer journeys (high ambient load) and spring/autumn journeys.

---

### Step 3: Dual Model Architecture

#### Engine A: Deterministic Physics Health-Index
The physics score represents direct domain thermodynamic knowledge:
$$S_{\text{phys}, c} = 1.0 \cdot z(\Delta T_{\text{rel}}) + 0.5 \cdot z(\Delta T_{\text{set}}) + 0.25 \cdot z(P95) + 0.25 \cdot z(\text{frac}_{>0.2}) + 0.25 \cdot z(\text{persist}_{15\text{m}})$$

#### Engine B: Pairwise Logistic Ranking Model
To learn subtle nonlinear interactions without overfitting, car comparisons are converted into a pairwise classification task:
- For every pair of active cars $(c_a, c_b)$ in a journey:
  $$\mathbf{x}_{a, b} = \mathbf{z}_a - \mathbf{z}_b$$
  $$y_{a, b} = \begin{cases} 1, & \text{if } c_a \text{ is the true faulty car} \\ 0, & \text{if } c_b \text{ is the true faulty car} \end{cases}$$
- A regularized **Logistic Regression** model with $L_2$ penalty ($C=0.20$) is fitted on the 76 decisive pairs generated across the 6 training cases.
- In inference, a round-robin tournament evaluates each car against all peers:
  $$\text{Pairwise\_Score}(c_a) = \frac{1}{N - 1} \sum_{b \neq a} P(c_a \succ c_b \mid \mathbf{x}_{a, b})$$

---

### Step 4: Hybrid Rank Aggregation & Cross-Validation

The final car score is a weighted blend:
$$\text{Score}_{\text{final}, c} = 0.75 \cdot Z(S_{\text{phys}, c}) + 0.25 \cdot Z(\text{Pairwise\_Score}(c))$$

#### Leave-One-Case-Out Cross-Validation (LOOCV) Results:
We systematically held out each of the 6 labeled cases, trained the pairwise model on the remaining 5, and evaluated ranking performance:

| Journey / Case | Total Cars | True Faulty Car | Predicted Rank 1 | Rank Assigned | Linear Rank-Decay Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Case 01** | 8 | Car 01 | Car 01 | **1st** | **1.0000** |
| **Case 02** | 8 | Car 04 | Car 04 | **1st** | **1.0000** |
| **Case 03** | 8 | Car 03 | Car 03 | **1st** | **1.0000** |
| **Case 04** | 4 | Car 03 | Car 03 | **1st** | **1.0000** |
| **Case 05** | 8 | Car 03 | Car 08 | **2nd** | **0.8750** |
| **Case 06** | 8 | Car 08 | Car 08 | **1st** | **1.0000** |
| **Overall Mean LOOCV** | — | — | — | — | **0.9583 / 1.0000** |

*Note on Case 05*: In Case 05, the true leaking car (Car 03) was assigned Rank 2, giving an individual case score of $7/8 = 0.8750$. In all other 5 cases, the model achieved perfect Top-1 localization ($1.0000$).

---

### Step 5: Test Case Inference & Stability Vetting

Evaluating the trained model on the unlabelled competition test case ([`acv_test_case.xlsx`](file:///c:/Users/Rald999/Documents/GitHub/nebulax_p3/PS3/02_Datasets/ACV/Test/acv_test_case.xlsx)):

```
Diagnosed Leaking Car : Car 01
Ranked String         : 01|04|03|08|07|06|02|05
Confidence Margin     : 1.338 (Large gap above 2nd ranked Car 04)
```

#### Diagnostic Breakdown for Test Case:
| Rank | Car | Final Score | Physics Z-Score | Mean $\Delta T_{\text{rel}}$ | Setpoint Deficit $\Delta T_{\text{set}}$ | Persistence (>15m) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Car 01** | **+1.98** | **+2.04** | **+0.82°C** | **+1.41°C** | **84.3%** |
| 2 | Car 04 | +0.64 | +0.61 | +0.21°C | +0.55°C | 31.2% |
| 3 | Car 03 | +0.29 | +0.25 | +0.08°C | +0.32°C | 18.0% |
| 4 | Car 08 | -0.11 | -0.14 | -0.05°C | +0.10°C | 5.2% |
| 5 | Car 07 | -0.38 | -0.35 | -0.14°C | -0.08°C | 0.0% |
| 6 | Car 06 | -0.62 | -0.59 | -0.22°C | -0.25°C | 0.0% |
| 7 | Car 02 | -0.85 | -0.88 | -0.31°C | -0.42°C | 0.0% |
| 8 | Car 05 | -0.95 | -0.94 | -0.39°C | -0.53°C | 0.0% |

#### Sensitivity / Perturbation Check:
Sweeping the ensemble blending weight $\alpha \in [0.50, 1.00]$ (varying between 50% physics and 100% pure physics) produced **identical rank order**, proving that the diagnosis of Car 01 is robust against model weighting variations.

---

## 3. Verification & Reproduction Instructions

To let any external vetting model or auditor verify the code and outputs:

### 1. Retrain and Export Model Bundle:
```bash
python ACV/export_acv_model_bundle.py
```
*Expected Output*: Fits pairwise model on 76 decisive pairs and exports `acv_model_bundle.joblib` (2.1 KB).

### 2. Run Verification Inference:
```bash
python -c "from acv_pipeline import ACVPredictor; p = ACVPredictor(); print(p.predict_file('PS3/02_Datasets/ACV/Test/acv_test_case.xlsx'))"
```
*Expected Output*: Output dictionary identifying Car 01 as rank 1 with ranking string `'01|04|03|08|07|06|02|05'`.

---

## 4. Key Questions for External Model Vetting

When submitting this dossier to another AI model or technical advisor, ask them to evaluate:
1. **Sample Size vs Architecture Choice**: Does the choice of a Physics-Informed Cross-Car Ranker + Pairwise Logistic Model appropriately mitigate the severe overfitting risk inherent in having only 6 training journeys?
2. **Sensor Set Uniformity**: Does the decision to exclude compressor discharge pressure and suction pressure from the primary model (since they are absent in `acv_test_case.xlsx` and Cases 01–03, 05, 06) maintain strict parity between train and inference?
3. **Data Cleaning Protocol**: Is the handling of CAN-bus zeros ($0.0^\circ\text{C}$ dropouts) and validity flags scientifically sound and consistent with rolling stock SCADA standards?
4. **Metric Alignment**: Is the consist-level round-robin pairwise ranking structure mathematically aligned with the competition's Linear Rank-Decay scoring function?
