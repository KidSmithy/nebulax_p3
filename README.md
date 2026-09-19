# 🚆 NebulaX P3 — Unified Rail Digital Twin

> **Interactive 3D Digital Twin & Predictive Maintenance System for Railway Fleet & Track Infrastructure**  
> Built for the **Land Transport Authority (LTA) NebulaX 2026 Hackathon — Problem Statement 3 (Train Condition Monitoring)**.

---

## 🌟 Overview

The **NebulaX P3 Unified Rail Digital Twin** is an operational condition-monitoring platform that unifies four heterogeneous railway subsystems into a single real-time vehicle-infrastructure operational environment:

1. **Door Subsystem (Cabin Envelope)**: Motor current signature analysis, kinematic obstruction detection, transit cycle duration, and guide-rail wear.
2. **Air Conditioning & Ventilation (ACV) (Auxiliary Climate Pack)**: Thermodynamic heat exchange, compressor power curves, COP efficiency, and refrigerant leakage localisation.
3. **Structural Health Monitoring (SHM) (Bogie & Running Gear)**: High-frequency axle-box acceleration ($g$), dynamic strain, weld joint fatigue, and bearing defect frequencies via ASTM E1049-85 Rainflow cycle counting and Welch PSD spectrum analysis.
4. **Rail Corrugation (Continuous Track Infrastructure)**: Surface roughness displacement ($\mu\text{m}$), spatial wavelength class (short-pitch vs. long-pitch), and grinding prioritization along Kilometre Post ($KP$) chainage.

### 🔬 Core Innovation: Physical Cross-Correlation
Track corrugation directly induces high-frequency shock loads into the wheelset, triggering elevated structural stress in the bogie bolster rib (`SHM`), while onboard auxiliary systems (`Door`, `ACV`) experience simultaneous operational duty cycles.

---

## 🏗️ Full-Stack 50/50 Architecture

```
+----------------------------------------------------------------------------------------------------+
|                                    UNIFIED OPERATIONAL BOUNDARY                                    |
+-------------------------------------------------+--------------------------------------------------+
| BACKEND: INGESTION, INFERENCE & STREAMING (50%) | FRONTEND: 3D SCENE, SHADERS & HUD (50%)          |
+-------------------------------------------------+--------------------------------------------------+
| 1. Data Ingestion & Spatial Harmonization       | 1. Multi-Scale 3D Scene Graph (R3F)              |
|    - 10 Hz temporal tick alignment (100ms)      |    - Hierarchical camera lerp (Macro/Meso/Micro) |
|    - Chainage mapping: KP(t) integration        |    - Procedural train car, bogie & track ribbon  |
| 2. Multi-Model Inference Pipeline               | 2. Dynamic GLSL Shaders & Kinematic Binding      |
|    - Door: Random Forest / Segment extraction   |    - Bogie FEA stress gradient vertex/frag shader|
|    - ACV: Pairwise Regressor & Thermal Delta    |    - Track corrugation displacement & stripes    |
|    - SHM: Rainflow / Welch PSD Fatigue Model    |    - Dual-leaf door kinematics & "ghost" mesh    |
|    - Rail: Axle-box XGBoost / Multi-class       |    - Volumetric ACV airflow particle vectors     |
| 3. Streaming Engine & Downsampler               | 3. High-Frequency Telemetry Cockpit (HUD)        |
|    - Full-duplex WebSocket server (10 Hz)       |    - Synchronized ECharts waveform monitors      |
|    - LTTB algorithm for waveform decimation     |    - Dark-mode glassmorphism HUD with Tailwind   |
| 4. Counterfactual "What-If" Engine              | 4. Interactive Simulation & Playback Sandbox     |
|    - Maintenance intervention simulator         |    - Timeline playback scrubber (1x - 10x, seek) |
|    - Closed-loop telemetry recalculation        |    - Maintenance sandbox intervention toggles    |
+-------------------------------------------------+--------------------------------------------------+
```

---

## ⚡ Key Capabilities

### 1. Multi-Scale 3D Scene (React Three Fiber & Drei)
- **Macro View**: Orbiting wide corridor showing the train progressing along the track ribbon and catenary alignment.
- **Meso View**: Isometric rolling stock car body with an interactive **X-Ray toggle** ($1.0 \rightarrow 0.25$ shell opacity) to expose interior assemblies.
- **Micro View**: Isolated camera zoom on targeted subsystems (Bogie running gear, Door 3R, or Rooftop ACV pack).

### 2. Custom Model-Driven GLSL Shaders
- **Bogie FEA Stress Shader (`FEAStressShader.ts`)**: Dynamically shifts vertex and fragment coloration based on real-time axle-box vibration RMS and weld node fatigue:
  - `Navy Blue (#1e3a8a)` $\rightarrow$ Nominal `Emerald (#059669)` $\rightarrow$ Elevated `Amber (#d97706)` $\rightarrow$ Critical `Crimson (#e11d48)`.
- **Corrugation Ribbon Shader (`CorrugationRibbonShader.ts`)**: Displaces track rail vertices according to surface roughness depth ($\mu\text{m}$) and projects dynamic amber hazard alert stripes across sections where severity $> 0.70$.

### 3. Kinematic Door Simulation with Ghost Baseline
- Dual-leaf kinematics driven by cycle state (`OPENING`, `CLOSING`, `DWELL`).
- Injects mechanical stuttering and jitter when `anomaly_score > 0.5`.
- Concurrently renders a translucent green wireframe **Ghost Mesh** representing nominal mechanical timing, visually highlighting the spatial lag $\Delta x$.

### 4. GPU Instanced Airflow Particles
- Simulates ACV climate pack airflow with warm amber cabin intake transitioning through cooling coils to bright cyan conditioned air, with stream velocity directly tied to compressor duty cycle.

### 5. High-Frequency Telemetry Cockpit (HUD)
- Synchronized **Apache ECharts** monitoring motor current signatures against nominal envelopes and axle-box FFT vibration spectra.
- **Interactive Timeline Scrubber**: Play/Pause, reset, speed multipliers ($1\times, 2\times, 5\times, 10\times$), and chainage slider.
- **Counterfactual "What-If" Sandbox**: Operator toggles (`ACTION_GRIND_RAIL`, `ACTION_LUBRICATE_DOOR`, `ACTION_REPLACE_FILTER`, `ACTION_INSPECT_BEARING`) with real-time recalculation of risk scores, 3D shaders, and chart waveforms.

---

## 📁 Repository Structure

```
nebulax_p3/
├── backend/
│   ├── api/
│   │   ├── routes.py                # REST API endpoints (/api/health, /api/whatif, etc.)
│   │   └── websocket_streamer.py    # 10 Hz full-duplex WebSocket server
│   ├── core/
│   │   ├── config.py                # Corridor geometry & threshold configurations
│   │   ├── harmonizer.py            # Temporal-spatial alignment into unified frames
│   │   └── seed_findings.py         # Replays saved model output as first-load findings
│   ├── models/
│   │   ├── door_model.py            # Door Random Forest classifier & waveform evaluator
│   │   ├── acv_model.py             # ACV thermodynamic regressor & leak detection
│   │   ├── shm_model.py             # Rainflow fatigue & Welch PSD vibration model
│   │   ├── corrugation_model.py     # Rail corrugation classification & grinding priority
│   │   └── inference_broker.py      # Multi-model orchestrator & Fleet Health Index
│   ├── services/
│   │   ├── downsampler.py           # LTTB decimation algorithm
│   │   └── whatif_engine.py         # Counterfactual simulation engine
│   ├── utils/
│   │   └── mock_generator.py        # Synchronized physical telemetry generator
│   ├── main.py                      # FastAPI application entry point
│   └── requirements.txt             # Python dependencies
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── canvas/
│   │   │   │   ├── TwinCanvas.tsx             # 3D Scene graph & camera lerp orchestrator
│   │   │   │   ├── TrainAssembly.tsx          # Procedural train car & bogies
│   │   │   │   ├── TrackCorridor.tsx          # Sleepers, catenary & rail ribbon
│   │   │   │   ├── kinematics/
│   │   │   │   │   ├── DoorAssembly.tsx       # Kinematic doors with jitter
│   │   │   │   │   └── GhostMesh.tsx          # Translucent nominal baseline
│   │   │   │   ├── particles/
│   │   │   │   │   └── AirflowParticles.tsx   # ACV airflow vector particles
│   │   │   │   └── shaders/
│   │   │   │       ├── FEAStressShader.ts     # Bogie stress colormap shader
│   │   │   │       └── CorrugationRibbonShader.ts # Rail corrugation shader
│   │   │   └── hud/
│   │   │       ├── CockpitHeader.tsx          # Top fleet telemetry bar & X-Ray toggle
│   │   │       ├── SubsystemSelector.tsx      # Subsystem cards & status badges
│   │   │       ├── TelemetryMonitor.tsx       # Synchronized ECharts monitors
│   │   │       ├── TimelineScrubber.tsx       # Playback controls & KP seek slider
│   │   │       └── WhatIfSandbox.tsx          # Counterfactual intervention drawer
│   │   ├── store/
│   │   │   └── useTwinStore.ts                # Global Zustand state management
│   │   ├── types/
│   │   │   └── telemetry.ts                   # TypeScript telemetry schemas
│   │   ├── App.tsx                            # Root application view
│   │   └── globals.css                        # Tailwind & glassmorphism theme
│   ├── package.json
│   ├── tailwind.config.js
│   └── vite.config.ts
│
├── PS3/                             # Hackathon Problem Statement 3 datasets & specifications
├── Door/                            # Trained Door subsystem model bundle & evaluation scripts
├── ACV/                             # Trained ACV model bundle & pipeline
├── Rail_Corrugation/                # Trained Rail corrugation model bundle & features
├── SHM/                             # Trained SHM model artifact (shm_model.pkl)
├── antigravity.config.yaml          # Project services configuration
└── antigravity_rail_digital_twin_spec.md # Master digital twin specification
```

---

## 🚀 Quick Start Guide

### Prerequisites
- **Python 3.11+** (virtual environment located in `./venv`)
- **Node.js 18+** and **npm**

### Step 1: Start the Backend Service

**Option A: From project root:**
```powershell
.\venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**Option B: From inside the `backend/` directory:**
```powershell
cd backend
..\venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# Or directly:
..\venv\Scripts\python.exe main.py
```
- **REST Health API**: [http://localhost:8000/api/health](http://localhost:8000/api/health)
- **Live WebSocket Stream**: `ws://localhost:8000/ws/telemetry`


### Step 2: Start the Frontend Application

```powershell
# Open a new terminal:
cd frontend
npm install
npm run dev
```
- Open [http://localhost:3000](http://localhost:3000) in any modern browser.

---

## 📡 Data Contract (`telemetry_frame.json`)

The backend broadcasts standardized 10 Hz JSON telemetry frames over WebSockets:

```json
{
  "frame_id": 14029,
  "timestamp": "2026-09-18T10:00:00.000Z",
  "track_chainage_km": 14.850,
  "train_speed_kmh": 68.4,
  "fleet_health_index": 0.86,
  "subsystems": {
    "door": {
      "active_door_id": "DOOR_3R",
      "cycle_state": "CLOSING",
      "transit_time_seconds": 3.42,
      "motor_current_amps": 11.85,
      "nominal_current_amps": 8.20,
      "anomaly_score": 0.78,
      "fault_type": "GUIDE_RAIL_FRICTION",
      "ghost_deviation_mm": 18.2,
      "waveform_window": [2.1, 4.3, 7.8, 11.85, 11.2, 5.0, 1.2]
    },
    "acv": {
      "unit_id": "ACV_PACK_1",
      "supply_temp_c": 19.2,
      "return_temp_c": 26.5,
      "delta_temp_c": 7.3,
      "compressor_power_kw": 4.82,
      "efficiency_rating": 0.71,
      "anomaly_score": 0.32,
      "fault_type": "NONE"
    },
    "shm": {
      "bogie_id": "BOGIE_FRONT",
      "vibration_rms_g": 3.82,
      "peak_frequency_hz": 142.5,
      "bearing_defect_prob": 0.24,
      "fatigue_damage_index": 0.64,
      "anomaly_score": 0.81,
      "critical_weld_node": "BOLSTER_RIB_L2",
      "fft_spectrum": [
        {"freq_hz": 25.0, "amp": 0.42},
        {"freq_hz": 142.5, "amp": 2.89},
        {"freq_hz": 300.0, "amp": 0.15}
      ]
    },
    "rail_corrugation": {
      "kp_start": 14.800,
      "kp_end": 14.900,
      "depth_microns": 42.0,
      "wavelength_class": "SHORT_PITCH",
      "severity_score": 0.89,
      "maintenance_urgency": "SCHEDULE_GRINDING_7D",
      "grinding_priority_rank": 2
    }
  },
  "active_interventions": {
    "ACTION_GRIND_RAIL": true,
    "ACTION_LUBRICATE_DOOR": false,
    "ACTION_REPLACE_FILTER": false,
    "ACTION_INSPECT_BEARING": false
  },
  "plain_status": {
    "shm.vibration_rms_g": "ACTION_NEEDED",
    "acv.efficiency_rating": "WATCH",
    "door.motor_current_amps": "GOOD"
  },
  "counterfactual": {
    "active": true,
    "active_actions": ["ACTION_GRIND_RAIL"],
    "headline": "Overall health improved from 35% to 91%",
    "improved_count": 6,
    "metrics": [
      {
        "path": "shm.vibration_rms_g",
        "label": "Bogie vibration",
        "unit": "g",
        "before": 4.78,
        "after": 2.17,
        "delta": -2.61,
        "percent_change": -54.6,
        "direction": "better",
        "status_before": "ACTION_NEEDED",
        "status_after": "GOOD"
      }
    ]
  },
  "next_corrugation_zone": {
    "kp_start": 18.250,
    "kp_centre": 18.300,
    "distance_km": 3.398,
    "is_inside_zone": false
  }
}
```

### Counterfactual block

Every frame is evaluated **twice from the same physical instant using the same
noise seed**: once with the operator's maintenance interventions applied, once
with all of them reverted. `counterfactual.metrics` is the measured difference
between those two evaluations, so the "what did my repair achieve?" figures in
the UI are real measurements rather than hardcoded estimates. When an action has
no effect, the delta is honestly zero and the UI explains why.

`plain_status` carries a `GOOD` / `WATCH` / `ACTION_NEEDED` verdict per metric,
derived from `METRIC_THRESHOLDS` in `backend/core/config.py`. The frontend
glossary mirrors those thresholds, so the words and colours in the HUD can never
contradict the backend's own judgement.

### WebSocket message kinds

| `type` | Direction | Purpose |
| --- | --- | --- |
| `frame` | server → client | 10 Hz telemetry frame, as above |
| `whatif_result` | server → client | Confirmation of a toggled intervention, including its measured impact |
| `playback` | client → server | Play/pause and speed multiplier |
| `seek` | client → server | Jump to a chainage |
| `whatif` | client → server | Apply or revert a maintenance action |
| `reset_whatif` | client → server | Revert every action at once |

---

## 🤖 AI Insight Layer (optional)

The HUD's **AI** tab turns the raw engineering readings into a plain-language
explanation for non-specialists, plus a free-text Q&A box.

### Setup

```powershell
copy backend\.env.example backend\.env
# then edit backend\.env and set your key
```

```ini
OPENAI_API_KEY=sk-proj-your-key-here
OPENAI_MODEL=gpt-5.6-terra
AI_INSIGHT_MIN_INTERVAL_SEC=6
```

Install the two extra dependencies (already in `requirements.txt`):

```powershell
.\venv\Scripts\python.exe -m pip install "openai>=1.99,<2" python-dotenv
```

### Behaviour and safeguards

- **The key never leaves the server.** The browser calls this backend; only the
  backend calls OpenAI.
- **The model is given pre-interpreted readings**, each annotated with its
  threshold and the verdict the UI is already displaying. It cannot invent its
  own idea of "normal" or contradict the dashboard.
- **Calls are throttled and cached** on a coarse fingerprint of the operational
  situation. A 10 Hz stream never becomes 10 model calls per second, and
  auto-refresh is opt-in at 20 s intervals.
- **Graceful degradation.** With no key, an exhausted quota, or an API error, a
  deterministic rule-based explanation is returned and labelled as such. The
  dashboard degrades in quality, never in availability.

### Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/ai/status` | Whether AI is enabled, and which model |
| `POST` | `/api/ai/insight` | Structured plain-language explanation of the current frame |
| `POST` | `/api/ai/ask` | Free-text question grounded in the live readings |
| `GET` | `/api/metrics/glossary` | Backend thresholds, for asserting UI/backend agreement |
| `GET` | `/api/whatif/actions` | Available interventions and their plain-language copy |
| `POST` | `/api/whatif/reset` | Revert all interventions |
| `POST` | `/api/predict/seed` | Restore the four pre-seeded demo findings |

---

## 🔖 Pre-seeded inspection tags (all four bubbles on first load)

Opening the site shows **all four inspection bubbles already on the NSL train**
— door, aircon, bogie and track — instead of an empty train that only gets tags
after four manual uploads.

Those numbers are **real model output, not mock values**. Inference is run once,
offline, over real files from `PS3/02_Datasets/`, and the unedited result
envelopes are saved to `backend/model_data/seed_findings.json`, which the
harmonizer replays at startup through the same `set_upload_result()` an upload
uses. The only "hardcoded" part is *which* dataset file was analysed — exactly
as if an operator had uploaded it a moment before you opened the page.

| Subsystem | Seeded dataset | Model verdict |
| --- | --- | --- |
| ACV | `ACV/Train/acv_case_05.xlsx` (labelled faulty car 04) | `WATCH` — car 04 leading suspect, ambiguous confidence |
| Door | `Door/Train.csv` (raw, **not** `_cleaned`) | `ACTION_NEEDED` — 30 of 110 cycles abnormal (27.3%) |
| SHM | `SHM/Train/train31.csv` (highest labelled damage, 0.928) | `ACTION_NEEDED` — inspect the axle-box bearing |
| Rail | `Rail_Corrugation/Train/Train2.csv` (labelled `Side II`) | `ACTION_NEEDED` — schedule grinding |

All four seeds come from the **Train** splits on purpose, so every `Test` file
stays unseen and available for the live upload walkthrough.

`acv_case_05` is deliberate: its faulty car (04) is the UI's `DEFAULT_CAR`, so
the camera does not jump on first load.

### ⚠️ Feed the door model `Train.csv`, never `Train_cleaned.csv`

`door_model.predict_from_csv` has its own column normaliser, but its alias table
targets the **original LTA headers** (`Motor current(mA)`, `Datetime`, …). The
`Door/*_cleaned.csv` files were normalised for a different (notebook) pipeline
and use snake_case (`motor_current_ma`, `datetime_str`, …), which that table does
not recognise:

| Input | Channels resolved | Cycle segmentation | Cycles found |
| --- | --- | --- | --- |
| `Train.csv` | **17 / 17** | real timestamp gaps | **110** (matches answer key) |
| `Train_cleaned.csv` | 0 / 17 | blind 150-row chunks | 121 (fabricated) |
| `Test.csv` | **17 / 17** | real timestamp gaps | 38 |
| `Test_cleaned.csv` | 0 / 17 | blind 150-row chunks | 42 (fabricated) |

With a cleaned file, motor current survives only by the "first numeric column"
fallback, the other 16 channels are silently replaced by **hardcoded constants**
(voltage 5000, EMF 500, leaf position 350), and `mean_duration_s` becomes an
artifact of the chunk size rather than a measurement. The verdict happens to
land in roughly the same place because current dominates the features — it is
accidentally right, not right.

Validated against `Door/Train_Segments_Answer.csv`: on raw `Train.csv` the model
reproduces the answer key exactly — **110/110 cycles, 30/30 abnormal, and 40/40
per-segment status and open/close agreement** over the returned segment detail.


Everything downstream behaves normally — the AI cards, the cockpit charts,
**Resolve** and the resolved log all treat these as ordinary findings, and the
seeded SHM result really does drive the generator's bearing wear.

### Regenerating, resetting and disabling

```powershell
# re-run the real models over the datasets and rewrite the saved JSON
.\venv\Scripts\python.exe -m backend.scripts.build_seed_findings

# only some subsystems
.\venv\Scripts\python.exe -m backend.scripts.build_seed_findings door shm

# put the bubbles back mid-demo, after resolving them (no restart needed)
curl -X POST http://localhost:8000/api/predict/seed
```

| Env var | Default | Effect |
| --- | --- | --- |
| `SEED_DEMO_FINDINGS` | `1` | Set to `0` for the original empty-until-you-upload start |
| `SEED_DEMO_LINES` | `NSL` | Comma-separated; e.g. `NSL,EWL` to seed both lines |

**EWL is intentionally left empty**, so the upload walkthrough still has a clean
line to demonstrate on. A missing or corrupt `seed_findings.json` degrades
silently to the old empty-state behaviour rather than failing startup — and the
datasets themselves are never needed at runtime.


---

## 👓 Making the dashboard readable for non-experts

The HUD ships a **Beginner/Expert toggle** (the `SIMPLE` / `EXPERT` button).

- **Beginner** replaces jargon with everyday wording — "Axle-box vibration RMS"
  becomes "Wheel shaking", "COP efficiency rating" becomes "Efficiency" — and
  states each verdict in words (`Normal`, `Watch`, `Act`) rather than relying on
  colour alone.
- **Expert** restores the engineering terminology and raw units.

Every reading carries a **?** tooltip defining what it measures, why it matters,
its healthy range, and an everyday analogy. Definitions live in one place,
`frontend/src/lib/metricGlossary.ts`, alongside a `TERM_GLOSSARY` that explains
enum values such as `SHORT_PITCH` or `GUIDE_RAIL_FRICTION`.

---

## 🛠️ Verification & Testing

- **Backend Smoke Test**:
  ```powershell
  .\venv\Scripts\python.exe -c "from backend.core.harmonizer import TelemetryHarmonizer; h = TelemetryHarmonizer(); print(h.generate_next_frame())"
  ```
- **LTTB Downsampler Test**:
  ```powershell
  .\venv\Scripts\python.exe -c "from backend.services.downsampler import lttb_downsample; import numpy as np; print(len(lttb_downsample(np.random.randn(1000), 50)))"
  ```
- **Frontend Production Build**:
  ```powershell
  cd frontend
  npm run build
  ```

---

## 📄 License
This project was developed for the Land Transport Authority (LTA) NebulaX 2026 Hackathon.
