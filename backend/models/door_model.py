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
import io
from sklearn.ensemble import RandomForestClassifier
from backend.core.config import ROOT_DIR, DOOR_DIR, DATASETS_DIR, MODEL_DATA_DIR
from backend.models.submission import door_timestamp, make_submission, merge_submissions

DOOR_CHANNELS = [
    'Motor current(mA)', 'Motor Voltage(10mV)', 'Motor electrodynamic force',
    'Door opening time(.1s)', 'Door closing time(.1s)', 'Close command',
    'Open command', 'DCSR', 'DCSL', 'DLSR', 'DLSL',
    'Door Opened', 'Door Locked', 'Door is opening', 'Door is closing',
    'Door leaf position'
]

BUNDLE_PATHS = [
    MODEL_DATA_DIR / "door_model_bundle.joblib",
    DOOR_DIR / "door_model_bundle.joblib",
]

class DoorPredictor:
    def __init__(self):
        self.bundle = None
        self.model = None
        self.expected_features = 210
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
        for path in BUNDLE_PATHS:
            if path.exists():
                try:
                    self.bundle = joblib.load(path)
                    if isinstance(self.bundle, dict):
                        self.model = self.bundle.get('model', self.bundle)
                        if 'feature_cols' in self.bundle:
                            self.feature_cols = self.bundle['feature_cols']
                    else:
                        self.model = self.bundle
                    self.expected_features = getattr(self.model, 'n_features_in_', 210)
                    print(f"[DoorPredictor] Successfully loaded model bundle from {path} (expected_features={self.expected_features})")
                    return
                except Exception as e:
                    print(f"[DoorPredictor] Failed to load cached bundle from {path}: {e}")

        if self.model is None:
            raise FileNotFoundError(
                f"[DoorPredictor] Could not load model bundle from any candidate path: "
                f"{[str(p) for p in BUNDLE_PATHS]}. Silent mock model fallback has been disabled."
            )





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

    def predict_from_csv(self, file_source, file_name: str = "door_data.csv") -> dict:
        """
        Runs batch/cycle-level inference on an uploaded Door CSV dataset.
        Extracts electromechanical features per transit segment and predicts faults
        (Normal vs. Abnormal resistance).
        """
        is_excel = str(file_name).lower().endswith(('.xlsx', '.xls'))
        if isinstance(file_source, (bytes, bytearray)):
            bio = io.BytesIO(file_source)
            df = pd.read_excel(bio) if is_excel else pd.read_csv(bio)
        elif isinstance(file_source, str) and "\n" in file_source:
            df = pd.read_csv(io.StringIO(file_source))
        elif isinstance(file_source, pd.DataFrame):
            df = file_source.copy()
        else:
            df = pd.read_excel(file_source) if is_excel else pd.read_csv(file_source)


        # Standardize column naming for all 16 Door channels
        col_map = {c.strip(): c for c in df.columns}
        for std, aliases in [
            ('Datetime', ['datetime', 'time', 'timestamp', 'Timestamp']),
            ('Motor current(mA)', ['motor_current', 'current_ma', 'motor current(mA)', 'Motor current']),
            ('Motor Voltage(10mV)', ['motor_voltage', 'voltage', 'Motor voltage(10mV)', 'Motor Voltage']),
            ('Motor electrodynamic force', ['emf', 'back_emf', 'Motor EMF']),
            ('Door opening time(.1s)', ['door_opening_time', 'opening_time', 'Door opening time(0.1 s)']),
            ('Door closing time(.1s)', ['door_closing_time', 'closing_time', 'Door closing time(0.1 s)']),
            ('Close command', ['close_command', 'Close Command', 'close_cmd']),
            ('Open command', ['open_command', 'Open Command', 'open_cmd']),
            ('DCSR', ['dcsr', 'DCSR_switch']),
            ('DCSL', ['dcsl', 'DCSL_switch']),
            ('DLSR', ['dlsr', 'DLSR_switch']),
            ('DLSL', ['dlsl', 'DLSL_switch']),
            ('Door Opened', ['door_opened', 'Opened', 'opened']),
            ('Door Locked', ['door_locked', 'Locked', 'locked']),
            ('Door is opening', ['door_is_opening', 'is_opening']),
            ('Door is closing', ['door_is_closing', 'is_closing']),
            ('Door leaf position', ['position', 'door_position', 'Door Position']),
        ]:
            if std not in df.columns:
                for a in aliases:
                    if a in col_map:
                        df[std] = df[col_map[a]]
                        break

        # Fallback columns if missing
        if 'Motor current(mA)' not in df.columns:
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                df['Motor current(mA)'] = df[num_cols[0]]
            else:
                raise ValueError("Uploaded CSV must contain numeric motor current measurements.")

        defaults = {
            'Motor Voltage(10mV)': 5000.0,
            'Motor electrodynamic force': 500.0,
            'Door opening time(.1s)': 30.0,
            'Door closing time(.1s)': 30.0,
            'Close command': 0.0,
            'Open command': 0.0,
            'DCSR': 0.0,
            'DCSL': 0.0,
            'DLSR': 0.0,
            'DLSL': 0.0,
            'Door Opened': 0.0,
            'Door Locked': 0.0,
            'Door is opening': 0.0,
            'Door is closing': 0.0,
            'Door leaf position': 350.0,
        }
        for c, def_val in defaults.items():
            if c not in df.columns:
                df[c] = def_val

        # Detect cycle segments
        segments = []
        if 'Datetime' in df.columns:
            try:
                sample_dt = str(df['Datetime'].dropna().iloc[0])
                if '-' in sample_dt and sample_dt.count('-') >= 5:
                    sp = df['Datetime'].astype(str).str.split('-', expand=True).astype(int)
                    parsed_dt = pd.to_datetime({
                        'year': sp[0], 'month': sp[1], 'day': sp[2],
                        'hour': sp[3], 'minute': sp[4], 'second': sp[5],
                        'microsecond': sp[6] * 1000 if sp.shape[1] > 6 else 0
                    })
                else:
                    parsed_dt = pd.to_datetime(df['Datetime'])
                df['parsed_dt'] = parsed_dt
                gaps = np.where(df['parsed_dt'].diff().dt.total_seconds() > 0.1)[0]
                starts = [0] + list(gaps)
                ends = [g - 1 for g in gaps] + [len(df) - 1]
                for s, e in zip(starts, ends):
                    if e - s >= 5:
                        segments.append((s, e))
            except Exception as dt_err:
                print(f"[DoorPredictor] Timestamp parsing note: {dt_err}. Segmenting by stroke...")

        if not segments:
            chunk_size = 150
            for s in range(0, len(df), chunk_size):
                e = min(s + chunk_size - 1, len(df) - 1)
                if e - s >= 10:
                    segments.append((s, e))

        records = []
        x_rows_210 = []
        for i, (s, e) in enumerate(segments):
            sub = df.iloc[s:e+1]
            cur = sub['Motor current(mA)'].values.astype(float)
            volt = sub['Motor Voltage(10mV)'].values.astype(float)
            emf = sub['Motor electrodynamic force'].values.astype(float)
            pos = sub['Door leaf position'].clip(0, 700).values.astype(float)

            pos_diff = pos[-1] - pos[0]
            inferred_op = 'Open' if pos_diff > 0 else 'Close'
            speed = np.abs(np.diff(pos) / 0.02) if len(pos) > 1 else np.array([0.0])
            power = (volt / 100.0) * (cur / 1000.0)

            if 'parsed_dt' in df.columns:
                dur = (df['parsed_dt'].iloc[e] - df['parsed_dt'].iloc[s]).total_seconds()
                start_time = door_timestamp(df['parsed_dt'].iloc[s])
                end_time = door_timestamp(df['parsed_dt'].iloc[e])
            else:
                dur = round(len(sub) * 0.02, 2)
                start_time = end_time = None

            rec = {
                'cycle_index': i + 1,
                'start_time': start_time,
                'end_time': end_time,
                'duration_s': dur,
                'n_rows': len(sub),
                'op_is_open': 1 if inferred_op == 'Open' else 0,
                'operation': inferred_op,
                'cur_mean': float(np.mean(cur)),
                'cur_std': float(np.std(cur)),
                'cur_min': float(np.min(cur)),
                'cur_max': float(np.max(cur)),
                'cur_median': float(np.median(cur)),
                'cur_q75': float(np.percentile(cur, 75)),
                'cur_q90': float(np.percentile(cur, 90)),
                'cur_rms': float(np.sqrt(np.mean(cur**2))),
                'cur_integral': float(np.sum(np.abs(cur) * 0.02)),
                'volt_mean': float(np.mean(volt)),
                'volt_std': float(np.std(volt)),
                'volt_min': float(np.min(volt)),
                'volt_max': float(np.max(volt)),
                'volt_q75': float(np.percentile(volt, 75)),
                'emf_mean': float(np.mean(emf)),
                'emf_std': float(np.std(emf)),
                'emf_min': float(np.min(emf)),
                'emf_max': float(np.max(emf)),
                'emf_q75': float(np.percentile(emf, 75)),
                'work_elec': float(np.sum(np.abs(power) * 0.02)),
                'power_max': float(np.max(np.abs(power))),
                'speed_mean': float(np.mean(speed)) if len(speed) > 0 else 0.0,
                'speed_max': float(np.max(speed)) if len(speed) > 0 else 0.0,
                'speed_std': float(np.std(speed)) if len(speed) > 0 else 0.0,
                'switch_dc_trans': float(np.sum(np.abs(np.diff(sub['DCSR'].values)))) if 'DCSR' in sub else 0.0,
                'switch_dl_trans': float(np.sum(np.abs(np.diff(sub['DLSR'].values)))) if 'DLSR' in sub else 0.0,
            }
            records.append(rec)

            # Extract exact 210 features across all 16 Door channels
            v = sub[DOOR_CHANNELS].to_numpy(dtype=float)
            v = np.nan_to_num(v, nan=0.0)
            stats = [v.mean(0), v.std(0), v.min(0), v.max(0),
                     np.quantile(v, .25, axis=0), np.median(v, axis=0),
                     np.quantile(v, .75, axis=0), np.sqrt((v ** 2).mean(0)),
                     np.abs(v).mean(0), v[0], v[-1], v[-1] - v[0],
                     np.abs(np.diff(v, axis=0)).sum(0)]
            feat_row = np.r_[np.concatenate(stats), dur, len(v)]
            x_rows_210.append(feat_row)

        if not records:
            raise ValueError("Could not extract valid door travel cycles from uploaded data.")

        if self.expected_features == 210 and len(x_rows_210) > 0:
            X = np.asarray(x_rows_210, dtype=float)
        else:
            X = np.array([[r.get(c, 0.0) for c in self.feature_cols] for r in records])

        preds = self.model.predict(X)
        probs = self.model.predict_proba(X) if hasattr(self.model, 'predict_proba') else np.zeros((len(X), 2))

        total_cycles = len(records)
        abnormal_count = int(np.sum(preds == 1))
        normal_count = total_cycles - abnormal_count
        fault_rate = float(abnormal_count / total_cycles)

        # Compute summary metrics
        mean_duration = round(float(np.mean([r['duration_s'] for r in records])), 2)
        mean_current_rms = round(float(np.mean([r['cur_rms'] for r in records])) / 1000.0, 2)  # Amps
        max_current_peak = round(float(np.max([r['cur_max'] for r in records])) / 1000.0, 2)  # Amps

        # Calibrated anomaly score (0.0 - 1.0)
        anomaly_score = round(float(np.clip(fault_rate * 1.1 + (0.15 if abnormal_count > 0 else 0.0), 0.04, 0.98)), 2)

        if abnormal_count == 0:
            status = "GOOD"
            fault_type = "NONE"
            action = "NONE"
            conductor_summary = (
                f"Evaluated {total_cycles} door operating cycles in '{file_name}'. "
                f"All {normal_count} cycles completed within nominal electromechanical parameters "
                f"(average transit time {mean_duration}s, mean RMS current {mean_current_rms}A). "
                f"No mechanical guide-rail obstruction or motor overload detected."
            )
        elif fault_rate <= 0.25:
            status = "WATCH"
            fault_type = "ROLLER_BEARING_WEAR"
            action = "ACTION_LUBRICATE_DOOR"
            conductor_summary = (
                f"Evaluated {total_cycles} door operating cycles in '{file_name}'. "
                f"Detected {abnormal_count} cycle(s) with elevated resistance ({fault_rate*100:.1f}% fault rate). "
                f"Peak motor current reached {max_current_peak}A with cycle transit averaging {mean_duration}s. "
                f"Early guide-rail friction or bearing wear developing. Recommend scheduling guide-rail lubrication."
            )
        else:
            status = "ACTION_NEEDED"
            fault_type = "GUIDE_RAIL_FRICTION"
            action = "ACTION_LUBRICATE_DOOR"
            conductor_summary = (
                f"High-severity warning for '{file_name}': {abnormal_count} of {total_cycles} door cycles "
                f"({fault_rate*100:.1f}%) exhibited abnormal mechanical resistance and motor overcurrent. "
                f"Peak motor current spiked to {max_current_peak}A. "
                f"Immediate maintenance required: lubricate guide rails and check door leaf alignment."
            )

        # Submission CSV covers every cycle, and only exists when the file had real timestamps.
        submission = None
        if all(r['start_time'] for r in records):
            submission = make_submission(
                "door_predictions.csv",
                ["start_time", "end_time", "prediction"],
                [[r['start_time'], r['end_time'], "Abnormal resistance" if preds[i] == 1 else "Normal"]
                 for i, r in enumerate(records)],
            )

        segment_details = []
        for i, r in enumerate(records[:40]):  # Cap returned rows for performance
            prob_abnormal = float(probs[i][1]) if probs.shape[1] > 1 else (1.0 if preds[i] == 1 else 0.0)
            segment_details.append({
                "cycle": r['cycle_index'],
                "operation": r['operation'],
                "duration_s": r['duration_s'],
                "current_rms_a": round(r['cur_rms'] / 1000.0, 2),
                "peak_current_a": round(r['cur_max'] / 1000.0, 2),
                "status": "Abnormal resistance" if preds[i] == 1 else "Normal",
                "fault_probability": round(prob_abnormal, 2),
            })

        return {
            "subsystem": "door",
            "file_name": file_name,
            "status": status,
            "verdict": status,
            "total_cycles": total_cycles,
            "normal_cycles": normal_count,
            "abnormal_cycles": abnormal_count,
            "fault_rate_pct": round(fault_rate * 100, 1),
            "anomaly_score": anomaly_score,
            "fault_type": fault_type,
            "recommended_action": action,
            "mean_duration_s": mean_duration,
            "mean_current_rms_a": mean_current_rms,
            "max_current_peak_a": max_current_peak,
            "conductor_summary": conductor_summary,
            "segments": segment_details,
            "submission": submission,
        }

    def predict_batch(self, file_list: list, batch_name: str = "door_batch.zip") -> dict:
        """
        Evaluates a batch of Door cycle test CSV files (e.g. from an uploaded ZIP).
        file_list: list of (file_name, file_bytes)
        """
        results = []
        for fname, fbytes in file_list:
            try:
                res = self.predict_from_csv(fbytes, fname)
                results.append(res)
            except Exception as e:
                print(f"[DoorPredictor] Error processing {fname} in batch: {e}")

        if not results:
            raise ValueError(f"Could not parse any valid Door CSV files from '{batch_name}'.")

        total_files = len(results)
        total_cycles = sum(r["total_cycles"] for r in results)
        total_abnormal = sum(r["abnormal_cycles"] for r in results)
        total_normal = total_cycles - total_abnormal
        overall_fault_rate = float(total_abnormal / total_cycles) if total_cycles > 0 else 0.0

        worst_item = max(results, key=lambda r: r["anomaly_score"])
        worst_file = worst_item["file_name"]
        anomalous_files = [r for r in results if r["status"] != "GOOD"]
        anomaly_count = len(anomalous_files)

        max_current = max(r["max_current_peak_a"] for r in results)
        mean_rms_current = round(float(np.mean([r["mean_current_rms_a"] for r in results])), 2)

        if worst_item["status"] == "ACTION_NEEDED" or overall_fault_rate > 0.25:
            verdict = "ACTION_NEEDED"
            action = "ACTION_LUBRICATE_DOOR"
            conductor_summary = (
                f"Batch evaluation of {total_files} door run files ({total_cycles} total cycles) in '{batch_name}': "
                f"Severe door mechanism resistance detected across {anomaly_count} run(s) ({total_abnormal} abnormal cycles, "
                f"{overall_fault_rate*100:.1f}% overall fault rate). "
                f"Worst-case dataset is '{worst_file}' with peak current spiking to {max_current}A. "
                f"Immediate mechanical guide rail cleaning, lubrication, and door leaf realignment required."
            )
        elif anomaly_count > 0 or overall_fault_rate > 0.0:
            verdict = "WATCH"
            action = "ACTION_LUBRICATE_DOOR"
            conductor_summary = (
                f"Batch evaluation of {total_files} door run files ({total_cycles} total cycles) in '{batch_name}': "
                f"Intermittent resistance observed in {anomaly_count} of {total_files} runs "
                f"({total_abnormal} anomalous cycles, {overall_fault_rate*100:.1f}% fault rate). "
                f"Guide rail lubrication recommended during next depot inspection."
            )
        else:
            verdict = "GOOD"
            action = "NONE"
            conductor_summary = (
                f"Batch evaluation of {total_files} door run files ({total_cycles} total cycles) in '{batch_name}': "
                f"All {total_cycles} door operating cycles performed nominally across all test files "
                f"(mean RMS current {mean_rms_current}A, peak {max_current}A). "
                f"Passenger door electromechanical actuation is fully nominal."
            )

        batch_breakdown = [
            {
                "file": r["file_name"],
                "total_cycles": r["total_cycles"],
                "abnormal_cycles": r["abnormal_cycles"],
                "fault_rate_pct": r["fault_rate_pct"],
                "peak_current_a": r["max_current_peak_a"],
                "verdict": r["verdict"],
                "fault_type": r["fault_type"]
            }
            for r in results
        ]

        return {
            "subsystem": "door",
            "file_name": batch_name,
            "is_batch": True,
            "status": verdict,
            "verdict": verdict,
            "total_files": total_files,
            "anomaly_count": anomaly_count,
            "total_cycles": total_cycles,
            "normal_cycles": total_normal,
            "abnormal_cycles": total_abnormal,
            "fault_rate_pct": round(overall_fault_rate * 100, 1),
            "worst_file": worst_file,
            "anomaly_score": worst_item["anomaly_score"],
            "fault_type": worst_item["fault_type"],
            "recommended_action": action,
            "max_current_peak_a": max_current,
            "mean_current_rms_a": mean_rms_current,
            "conductor_summary": conductor_summary,
            "batch_items": batch_breakdown,
            "submission": merge_submissions(results),
        }


