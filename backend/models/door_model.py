"""
backend/models/door_model.py
===============================================================================
Door Subsystem Inference Engine:
- Cycle transit duration & motor current signature analysis
- Mechanical guide-rail friction & obstruction detection
- Anomaly score computation and kinematic ghost deviation
===============================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from backend.core.config import ROOT_DIR, DOOR_DIR, DATASETS_DIR

BUNDLE_PATH = DOOR_DIR / "door_model_bundle.joblib"

class DoorPredictor:
    def __init__(self):
        self.bundle = None
        self.model = None
        self.feature_cols = [
            'duration_s', 'n_rows', 'op_is_open',
            'cur_mean', 'cur_std', 'cur_min', 'cur_max', 'cur_median', 'cur_q75', 'cur_q90', 'cur_rms', 'cur_integral',
            'volt_mean', 'volt_std', 'volt_min', 'volt_max', 'volt_q75',
            'emf_mean', 'emf_std', 'emf_min', 'emf_max', 'emf_q75',
            'work_elec', 'power_max', 'speed_mean', 'speed_max', 'speed_std',
            'switch_dc_trans', 'switch_dl_trans'
        ]
        self._load_or_train()

    def _load_or_train(self):
        if BUNDLE_PATH.exists():
            try:
                self.bundle = joblib.load(BUNDLE_PATH)
                self.model = self.bundle['model']
                return
            except Exception as e:
                print(f"[DoorPredictor] Failed to load cached bundle: {e}. Retraining...")

        # Train baseline if bundle not found
        train_path = DATASETS_DIR / "Door" / "Train.csv"
        ans_path = DATASETS_DIR / "Door" / "Train_Segments_Answer.csv"
        if not train_path.exists():
            train_path = DOOR_DIR / "Train.csv"
            ans_path = DOOR_DIR / "Train_Segments_Answer.csv"

        if train_path.exists() and ans_path.exists():
            self._train_and_save(train_path, ans_path)
        else:
            # Fallback mock model
            self.model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            dummy_X = np.random.randn(20, len(self.feature_cols))
            dummy_y = np.random.randint(0, 2, size=20)
            self.model.fit(dummy_X, dummy_y)

    def _train_and_save(self, train_path, ans_path):
        df_train = pd.read_csv(train_path)
        df_ans = pd.read_csv(ans_path)

        # Parse timestamps
        sp = df_train['Datetime'].str.split('-', expand=True).astype(int)
        parsed_dt = pd.to_datetime({
            'year': sp[0], 'month': sp[1], 'day': sp[2],
            'hour': sp[3], 'minute': sp[4], 'second': sp[5],
            'microsecond': sp[6] * 1000
        })
        df_train['parsed_dt'] = parsed_dt

        # Detect segment gaps
        gaps = np.where(df_train['parsed_dt'].diff().dt.total_seconds() > 0.1)[0]
        starts = [0] + list(gaps)
        ends = [g - 1 for g in gaps] + [len(df_train) - 1]

        records = []
        for i in range(min(len(starts), len(df_ans))):
            s, e = starts[i], ends[i]
            sub = df_train.iloc[s:e+1]
            cur = sub['Motor current(mA)'].values
            volt = sub['Motor Voltage(10mV)'].values
            emf = sub['Motor electrodynamic force'].values
            pos = sub['Door leaf position'].clip(0, 700).values

            pos_diff = pos[-1] - pos[0]
            inferred_op = 'Open' if pos_diff > 0 else 'Close'
            speed = np.abs(np.diff(pos) / 0.02) if len(pos) > 1 else np.array([0.0])
            power = (volt / 100.0) * (cur / 1000.0)
            duration_s = (parsed_dt.iloc[e] - parsed_dt.iloc[s]).total_seconds()

            rec = {
                'duration_s': duration_s,
                'n_rows': len(sub),
                'op_is_open': 1 if inferred_op == 'Open' else 0,
                'cur_mean': np.mean(cur),
                'cur_std': np.std(cur),
                'cur_min': np.min(cur),
                'cur_max': np.max(cur),
                'cur_median': np.median(cur),
                'cur_q75': np.percentile(cur, 75),
                'cur_q90': np.percentile(cur, 90),
                'cur_rms': np.sqrt(np.mean(cur**2)),
                'cur_integral': np.sum(np.abs(cur) * 0.02),
                'volt_mean': np.mean(volt),
                'volt_std': np.std(volt),
                'volt_min': np.min(volt),
                'volt_max': np.max(volt),
                'volt_q75': np.percentile(volt, 75),
                'emf_mean': np.mean(emf),
                'emf_std': np.std(emf),
                'emf_min': np.min(emf),
                'emf_max': np.max(emf),
                'emf_q75': np.percentile(emf, 75),
                'work_elec': np.sum(np.abs(power) * 0.02),
                'power_max': np.max(np.abs(power)),
                'speed_mean': np.mean(speed) if len(speed) > 0 else 0,
                'speed_max': np.max(speed) if len(speed) > 0 else 0,
                'speed_std': np.std(speed) if len(speed) > 0 else 0,
                'switch_dc_trans': np.sum(np.abs(np.diff(sub['DCSR'].values))) if 'DCSR' in sub else 0,
                'switch_dl_trans': np.sum(np.abs(np.diff(sub['DLSR'].values))) if 'DLSR' in sub else 0,
                'target': 1 if df_ans['status'].iloc[i] == 'Abnormal resistance' else 0
            }
            records.append(rec)

        df_feat = pd.DataFrame(records)
        X = df_feat[self.feature_cols].values
        y = df_feat['target'].values

        self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.model.fit(X, y)
        self.bundle = {'model': self.model, 'feature_cols': self.feature_cols}
        joblib.dump(self.bundle, BUNDLE_PATH)
        print(f"[DoorPredictor] Model trained on {len(X)} cycles and saved to {BUNDLE_PATH}")

    def evaluate_cycle(self, motor_current_amps: float, nominal_current_amps: float,
                       transit_time: float, is_opening: bool = False,
                       is_moving: bool = True, nominal_transit_time: float = 3.05,
                       rng: np.random.Generator = None) -> dict:
        """
        Evaluates real-time kinematic and electromechanical door cycle.
        Returns anomaly_score, fault_type, ghost_deviation_mm, and waveform_window.

        Anomaly is judged on the RATIO of measured to expected current for the
        phase the door is currently in. While the door is parked (dwell or
        locked) the motor is unloaded, so no friction verdict can be made and
        the score is held at zero rather than dividing by a near-zero nominal.
        """
        if rng is None:
            rng = np.random.default_rng()

        peak_nominal = max(nominal_current_amps, 0.1)

        if not is_moving:
            # Door is parked. Nothing to diagnose from an unloaded motor.
            return {
                "active_door_id": "DOOR_3R",
                "cycle_state": "OPENING" if is_opening else "CLOSING",
                "transit_time_seconds": round(float(transit_time), 2),
                "motor_current_amps": round(float(motor_current_amps), 2),
                "nominal_current_amps": round(float(peak_nominal), 2),
                "anomaly_score": 0.0,
                "fault_type": "NONE",
                "ghost_deviation_mm": 0.0,
                "waveform_window": [round(float(motor_current_amps), 2)] * 10,
            }

        current_ratio = motor_current_amps / peak_nominal
        time_excess = max(0.0, transit_time - nominal_transit_time)

        # Baseline anomaly score (0.0 - 1.0)
        anomaly_score = float(np.clip((current_ratio - 1.0) * 1.6 + time_excess * 0.45, 0.0, 0.99))

        fault_type = "NONE"
        ghost_deviation_mm = 0.0

        if anomaly_score > 0.65:
            fault_type = "GUIDE_RAIL_FRICTION"
            ghost_deviation_mm = round(float((current_ratio - 1.0) * 22.5 + time_excess * 8.0), 1)
        elif anomaly_score > 0.40:
            fault_type = "ROLLER_BEARING_WEAR"
            ghost_deviation_mm = round(float((current_ratio - 1.0) * 12.0), 1)

        # Generate realistic 10-point waveform window scaled to the measured peak
        base_curve = np.array([0.22, 0.48, 0.88, 1.00, 0.94, 0.72, 0.48, 0.30, 0.15, 0.05])
        waveform = base_curve * motor_current_amps
        # Add slight jitter if anomalous (mechanical stutter / stick-slip)
        if anomaly_score > 0.5:
            jitter = rng.normal(0, 0.6 * anomaly_score, size=len(waveform))
            waveform = np.clip(waveform + jitter, 0.1, None)

        return {
            "active_door_id": "DOOR_3R",
            "cycle_state": "OPENING" if is_opening else "CLOSING",
            "transit_time_seconds": round(float(transit_time), 2),
            "motor_current_amps": round(float(motor_current_amps), 2),
            "nominal_current_amps": round(float(peak_nominal), 2),
            "anomaly_score": round(anomaly_score, 2),
            "fault_type": fault_type,
            "ghost_deviation_mm": ghost_deviation_mm,
            "waveform_window": [round(float(w), 2) for w in waveform]
        }
