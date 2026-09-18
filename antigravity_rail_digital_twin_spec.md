---
project: "NebulaX P3 Unified Rail Digital Twin"
target_agent: "Google Antigravity"
version: "1.0.0"
architecture: "Hybrid Vehicle-Infrastructure Digital Twin"
division_of_work:
  backend: "50% (Data Harmonization, Multi-Model Inference Engine, WebSocket Streamer, Downsampling, Counterfactual Simulator)"
  frontend: "50% (React Three Fiber 3D Rigging, Dynamic GLSL Shaders, Kinematic Ghosting, ECharts HUD, Zustand State)"
stack:
  backend:
    runtime: "Python 3.11+"
    frameworks: ["FastAPI", "WebSockets", "Uvicorn"]
    processing_and_ml: ["Polars", "NumPy", "ONNX Runtime", "PyTorch", "Scikit-Learn"]
  frontend:
    framework: "Next.js 14+ (App Router) / Vite React"
    graphics: ["Three.js", "@react-three/fiber", "@react-three/drei"]
    styling_and_ui: ["Tailwind CSS", "shadcn/ui", "Lucide React"]
    charts: ["Apache ECharts", "echarts-for-react"]
    state_management: ["Zustand"]
---

# Unified Rail Digital Twin: Technical Specification & Antigravity Build Directives

## 1. System Overview
This specification details the end-to-end architecture and implementation blueprint for building a unified, interactive 3D digital twin for railway predictive maintenance. The twin synthesizes four heterogeneous subsystems into a single cohesive vehicle-track operational environment:
1. **Door Subsystem** (Rolling stock cabin envelope): Motor current signature analysis, kinematic obstruction, cycle transit duration, and mechanical guide-rail wear.
2. **Air Conditioning & Ventilation (ACV)** (Auxiliary climate pack): Thermodynamic heat exchange, compressor power curves, and refrigerant leakage detection.
3. **Structural Health Monitoring (SHM)** (Bogie & running gear): High-frequency axle-box acceleration ($g$), dynamic strain, weld joint fatigue, and bearing defect frequencies.
4. **Rail Corrugation** (Continuous track infrastructure): Surface roughness displacement ($\mu\text{m}$), spatial wavelength class (short-pitch vs. long-pitch), and grinding prioritization along Kilometre Post ($KP$) chainage.

The core value of this digital twin is **physical cross-correlation**: track corrugation directly induces high-frequency dynamic loads into the wheelset, triggering elevated structural stress in the bogie (SHM), while onboard auxiliary systems (Door, ACV) experience simultaneous operational duty cycles.

---

## 2. 50/50 Full-Stack Architecture Breakdown

```
+----------------------------------------------------------------------------------------------------+
|                                    UNIFIED OPERATIONAL BOUNDARY                                    |
+-------------------------------------------------+--------------------------------------------------+
| BACKEND: INGESTION, INFERENCE & STREAMING (50%) | FRONTEND: 3D SCENE, SHADERS & HUD (50%)          |
+-------------------------------------------------+--------------------------------------------------+
| 1. Data Ingestion & Spatial Harmonization       | 1. Multi-Scale 3D Scene Graph (R3F)              |
|    - Temporal alignment ($10\,\text{Hz}$)       |    - Hierarchical camera lerp (Macro/Meso/Micro) |
|    - Chainage mapping ($KP(t)$ integration)     |    - Low-poly train, bogie & track ribbon meshes |
| 2. Multi-Model Inference Pipeline               | 2. Dynamic GLSL Shaders & Kinematic Binding      |
|    - Door: Random Forest / Isolation Forest     |    - Bogie FEA stress gradient vertex shader     |
|    - ACV: Thermodynamic Regressor / Autoencoder |    - Track corrugation displacement & roughness  |
|    - SHM: 1D-CNN / FFT Peak Classifier          |    - Asynchronous door leaf transit & "ghost"    |
|    - Rail: Spatial Wavelength Wavelet Model     |    - Volumetric ACV airflow particle vectors     |
| 3. Streaming Engine & Downsampler               | 3. High-Frequency Telemetry Cockpit (HUD)        |
|    - Full-duplex WebSocket server               |    - Synchronized ECharts waveform monitors      |
|    - LTTB algorithm for waveform decimation     |    - Responsive dark-mode glassmorphism HUD      |
| 4. Counterfactual "What-If" Engine              | 4. Interactive Simulation & Playback Sandbox     |
|    - Maintenance intervention simulator         |    - Timeline playback scrubber ($1\times-10\times$) |
|    - Closed-loop telemetry recalculation        |    - Maintenance sandbox intervention toggles    |
+-------------------------------------------------+--------------------------------------------------+
```

---

## 3. Backend Technical Directives (50%)

### 3.1. Module Responsibilities
The backend must be structured under `backend/` and execute as an asynchronous FastAPI service:

* **`backend/core/harmonizer.py`**:
  * Ingests asynchronous sensor streams across Door ($100\,\text{Hz}$ current profiles during transit), ACV ($1\,\text{Hz}$ thermodynamic temperatures), SHM ($1\,\text{kHz}$ accelerometer vibration), and Track Geometry ($1\,\text{m}$ spatial intervals).
  * Harmonizes signals into a unified temporal-spatial frame broadcast at $10\,\text{Hz}$ ($100\,\text{ms}$ ticks) indexed by both ISO-8601 UTC timestamp and continuous railway chainage $KP(t) = KP_0 + \int_0^t v(\tau)\,d\tau$.
* **`backend/models/inference_broker.py`**:
  * Manages concurrent model execution across all 4 domains using ONNX Runtime / PyTorch.
  * Produces standardized anomaly metrics: normalized severity score ($0.0 - 1.0$), confidence intervals, RUL predictions, and top-3 feature attributions.
* **`backend/services/downsampler.py`**:
  * Implements Largest-Triangle-Three-Buckets (LTTB) decimation to downsample raw high-frequency waveforms (e.g., $1000\,\text{Hz}$ vibration down to 100 points per window) without clipping critical harmonic peaks before sending over WebSockets.
* **`backend/services/whatif_engine.py`**:
  * Exposes counterfactual simulation endpoints. When an operator triggers a digital action (e.g., `ACTION_GRIND_RAIL`, `ACTION_LUBRICATE_DOOR`), this engine injects baseline parameters, recomputes synthetic telemetry, and immediately returns the revised risk projection across all affected models.
* **`backend/api/websocket_streamer.py`**:
  * Manages active client connections, handles client timeline seek requests, and broadcasts synchronized telemetry frames.

### 3.2. Data Contracts & Schemas

#### Unified Telemetry Frame (`telemetry_frame.json`)
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
  }
}
```

---

## 4. Frontend Technical Directives (50%)

### 4.1. Module Responsibilities
The frontend must be structured under `frontend/` using React Three Fiber and Next.js:

* **`frontend/src/components/canvas/TwinCanvas.tsx`**:
  * Orchestrates the 3D scene, cameras, environmental lighting, and camera lerp transitions.
  * Implements smooth camera animations between three hierarchical presets:
    * **Macro View**: Orbiting wide corridor showing the train progressing along the track ribbon.
    * **Meso View**: Mid-range isometric view of the rolling stock body with transparent X-Ray inspection toggle.
    * **Micro View**: Tight zoom on isolated assemblies (Bogie, Door 3R, or Roof HVAC pack).
* **`frontend/src/components/canvas/shaders/FEAStressShader.ts`**:
  * Custom Three.js vertex and fragment shader applied to the bogie mesh.
  * Takes `uniform float u_stress_intensity` and `uniform vec3 u_hotspot_coords`.
  * Computes a dynamic color ramp: Low stress (Navy Blue `#1e3a8a`), Nominal (Emerald `#059669`), Elevated (Amber `#d97706`), Critical (Hot Magenta/Crimson `#e11d48`).
* **`frontend/src/components/canvas/shaders/CorrugationRibbonShader.ts`**:
  * Applied to the track rail head ribbon.
  * Displaces vertices dynamically based on spatial depth ($\mu\text{m}$) and applies warning stripes on sections where severity $> 0.70$.
* **`frontend/src/components/canvas/kinematics/DoorAssembly.tsx`**:
  * Kinematic dual-leaf door model. Animates open/close transitions driven by `cycle_state` and `transit_time_seconds`.
  * When `anomaly_score > 0.5`, introduces mechanical stuttering/jitter in the animation curve.
  * Renders a translucent green wireframe "Ghost Baseline" mesh concurrently to visually reveal the spatial lag $\Delta x$ against nominal mechanical timing.
* **`frontend/src/components/canvas/particles/AirflowParticles.tsx`**:
  * GPU instanced particle system simulating ACV climate pack airflow.
  * Particles transition from deep amber (hot cabin intake) through evaporator coils to bright cyan (conditioned supply air).
* **`frontend/src/components/hud/TelemetryMonitor.tsx`**:
  * High-frequency dashboard rendering synchronized ECharts for motor current waveforms, FFT vibration spectrums, and thermal deltas.
* **`frontend/src/components/hud/TimelineScrubber.tsx`**:
  * Temporal playback controls: Play, Pause, Reverse, $1\times$, $2\times$, $5\times$, $10\times$ speeds, and timeline seek slider.
* **`frontend/src/components/hud/WhatIfSandbox.tsx`**:
  * Interactive action drawer. Provides trigger buttons: `Simulate Rail Grinding`, `Lubricate Door Guides`, `Replace ACV Air Filter`, and `Inspect Front Bearing`.
  * Dispatches counterfactual mutations to the backend and renders before/after risk deltas.
* **`frontend/src/store/useTwinStore.ts`**:
  * Global Zustand store managing active frame data, camera target, selected subsystem, playback state, and active counterfactuals.

---

## 5. Complete File Directory Tree

```
rail-digital-twin/
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   └── websocket_streamer.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── harmonizer.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── door_model.py
│   │   ├── acv_model.py
│   │   ├── shm_model.py
│   │   ├── corrugation_model.py
│   │   └── inference_broker.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── downsampler.py
│   │   └── whatif_engine.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── mock_generator.py
│   ├── main.py
│   └── requirements.txt
│
├── frontend/
│   ├── public/
│   │   └── models/
│   │       ├── train_car.glb
│   │       ├── bogie.glb
│   │       └── door_leaf.glb
│   ├── src/
│   │   ├── app/
│   │   │   ├── globals.css
│   │   │   ├── layout.tsx
│   │   │   └── page.tsx
│   │   ├── components/
│   │   │   ├── canvas/
│   │   │   │   ├── TwinCanvas.tsx
│   │   │   │   ├── TrainAssembly.tsx
│   │   │   │   ├── TrackCorridor.tsx
│   │   │   │   ├── kinematics/
│   │   │   │   │   ├── DoorAssembly.tsx
│   │   │   │   │   └── GhostMesh.tsx
│   │   │   │   ├── particles/
│   │   │   │   │   └── AirflowParticles.tsx
│   │   │   │   └── shaders/
│   │   │   │       ├── FEAStressShader.ts
│   │   │   │       └── CorrugationRibbonShader.ts
│   │   │   └── hud/
│   │   │       ├── CockpitHeader.tsx
│   │   │       ├── SubsystemSelector.tsx
│   │   │       ├── TelemetryMonitor.tsx
│   │   │       ├── TimelineScrubber.tsx
│   │   │       └── WhatIfSandbox.tsx
│   │   ├── store/
│   │   │   └── useTwinStore.ts
│   │   └── types/
│   │       └── telemetry.ts
│   ├── package.json
│   ├── tailwind.config.js
│   └── tsconfig.json
│
└── antigravity.config.yaml
```

---

## 6. Antigravity Agent Execution Checklist & Verification Gates

The AI agent must follow these sequential verification gates during project synthesis:

### Gate 1: Telemetry Contract & Ingestion Pipeline
- [ ] Implement `backend/utils/mock_generator.py` generating realistic synchronized data for Door, ACV, SHM, and Rail Corrugation.
- [ ] Implement `backend/core/harmonizer.py` and verify that time-space indexing aligns timestamps with continuous chainage ($KP$).
- [ ] Implement `backend/services/downsampler.py` (LTTB decimation) and verify that $1000\,\text{Hz}$ signals compress to $100$ points without losing peak vibration harmonics.
- [ ] Establish WebSocket broadcast on `ws://localhost:8000/ws/telemetry` at $10\,\text{Hz}$.

### Gate 2: 3D Scene Graph & Mesh Loading
- [ ] Scaffold Next.js frontend and install `@react-three/fiber`, `@react-three/drei`, `three`, `zustand`, `lucide-react`, and `echarts-for-react`.
- [ ] Build `TwinCanvas.tsx` with orbit controls, realistic metallic materials, and procedural track corridor ribbon.
- [ ] Implement camera lerp navigation smoothly transitioning between `Macro` (Track), `Meso` (Car Body), and `Micro` (Subsystems).
- [ ] Add an X-Ray view toggle modifying car body shell opacity ($1.0 \rightarrow 0.25$) to expose internal bogies and actuators.

### Gate 3: Model-Driven Shaders & Kinematics
- [ ] Bind `FEAStressShader.ts` to the bogie mesh. Verify that changing `shm.anomaly_score` dynamically transitions the shader from blue to deep magenta.
- [ ] Bind `CorrugationRibbonShader.ts` to rail coordinates. Verify that corrugated track segments display wave ripples and amber alert bands at designated KP markers.
- [ ] Bind `DoorAssembly.tsx` to door cycle telemetry. Verify mechanical jitter during high anomaly states and render the translucent green `GhostMesh.tsx` to display spatial delay.
- [ ] Bind `AirflowParticles.tsx` atop the ACV unit with dynamic velocity reflecting fan duty cycle.

### Gate 4: Closed-Loop Telemetry Cockpit & What-If Simulation
- [ ] Build `TelemetryMonitor.tsx` displaying real-time ECharts synchronized with active 3D playback.
- [ ] Connect `TimelineScrubber.tsx` to the backend WebSocket seek handler to allow scrubbing through time.
- [ ] Implement `WhatIfSandbox.tsx`. Clicking "Simulate Rail Grinding" must post to `backend/services/whatif_engine.py`, clearing the corrugation alert and instantly dropping bogie vibration in both the 3D shader and HUD charts.
