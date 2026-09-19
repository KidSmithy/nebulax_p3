"""
backend/models/corrugation_model.py
===============================================================================
Rail Corrugation Subsystem Model:
- Evaluates continuous track surface roughness and axle-box multi-channel shock/vibration
- Classifies Normal vs. Side I vs. Side II corrugation
- Computes roughness depth (microns), severity score, and grinding priority rank
===============================================================================
"""

import os
import joblib
import numpy as np
import io
import pandas as pd
from scipy.signal import welch
from backend.core.config import RAIL_DIR, PS3_DIR, MODEL_DATA_DIR, CORRUGATION_ZONES

BUNDLE_PATHS = [
    MODEL_DATA_DIR / "rail_model_bundle.joblib",
    RAIL_DIR / "models" / "rail_model_bundle.joblib",
    PS3_DIR / "models" / "rail_model_bundle.joblib"
]

BANDS = ((0, 100), (100, 300), (300, 800), (800, 1600), (1600, 3000), (3000, 5000))
DEFAULT_LABELS = ['Normal', 'Side I', 'Side II']

class RailCorrugationModel:
    def __init__(self):
        self.bundle = None
        self.models = []
        self.feature_names = []
        self.feature_indices = []
        self.class_labels = DEFAULT_LABELS
        self.model_id = "r3_cat_sqrt_random2"
        self._load_bundle()

    def _load_bundle(self):
        for p in BUNDLE_PATHS:
            if p.exists():
                try:
                    self.bundle = joblib.load(p)
                    if isinstance(self.bundle, dict):
                        self.model_id = self.bundle.get('model_id', self.model_id)
                        raw_models = self.bundle.get('models', [])
                        if raw_models and isinstance(raw_models[0], dict) and 'bundle' in raw_models[0]:
                            self.models = [m['bundle']['estimator'] for m in raw_models]
                            sub = raw_models[0]['bundle']
                            self.feature_names = sub.get('feature_names', [])
                            self.feature_indices = sub.get('feature_indices', [])
                            self.class_labels = sub.get('class_labels', DEFAULT_LABELS)
                        elif 'model' in self.bundle:
                            self.models = [self.bundle['model']]
                            self.feature_names = self.bundle.get('feature_cols', [])
                        else:
                            self.models = [m for m in raw_models if hasattr(m, 'predict_proba') or hasattr(m, 'predict')]

                        print(f"[RailCorrugationModel] Loaded model bundle from {p} with {len(self.models)} estimators")
                        return
                    else:
                        self.models = [self.bundle]
                        print(f"[RailCorrugationModel] Loaded single estimator from {p}")
                        return
                except Exception as e:
                    print(f"[RailCorrugationModel] Warning loading bundle from {p}: {e}")

        if not self.models:
            raise FileNotFoundError(
                f"[RailCorrugationModel] Could not load any valid corrugation model estimators from "
                f"{[str(p) for p in BUNDLE_PATHS]}."
            )

    def _extract_features_252(self, df: pd.DataFrame) -> np.ndarray:
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 129 or (len(num_cols) > 0 and 'speed' in str(num_cols[0]).lower()):
            x = df[num_cols[1:]].to_numpy(dtype=float)
        else:
            x = df[num_cols].to_numpy(dtype=float)

        if x.shape[1] != 128:
            return None

        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        if len(x) < 512:
            pad = np.zeros((512 - len(x), 128))
            x = np.vstack([x, pad])

        rms = np.sqrt(np.mean(x * x, axis=0))
        std = x.std(axis=0)
        absx = np.abs(x)
        quantiles = np.quantile(absx, (0.5, 0.95, 0.99), axis=0)
        centered = x - x.mean(axis=0)
        kurt = np.mean(centered ** 4, axis=0) / np.maximum(std ** 4, 1e-24)
        crest = np.max(absx, axis=0) / np.maximum(rms, 1e-12)

        nperseg = min(1024, len(x))
        freq, density = welch(x, fs=10000.0, window='hann', nperseg=nperseg,
                              noverlap=nperseg // 2, detrend='constant', scaling='density', axis=0)
        dfreq = freq[1] - freq[0] if len(freq) > 1 else 1.0
        power = density * dfreq
        total = power.sum(axis=0)
        shape = power / np.maximum(total, 1e-24)

        channel_stats = {
            'rms': rms, 'std': std, 'abs_q50': quantiles[0], 'abs_q95': quantiles[1],
            'abs_q99': quantiles[2], 'crest': crest, 'kurtosis': kurt,
            'spectral_centroid_hz': (shape * freq[:, None]).sum(axis=0),
            'spectral_entropy': -(shape * np.log(np.maximum(shape, 1e-24))).sum(axis=0) / np.log(max(2, len(freq))),
        }

        for low, high in BANDS:
            mask = (freq >= low) & (freq <= high if high == 5000 else freq < high)
            band_power = power[mask].sum(axis=0) if np.any(mask) else np.zeros(128)
            channel_stats[f'log1p_band_power_{low}_{high}hz'] = np.log1p(band_power)
            channel_stats[f'band_fraction_{low}_{high}hz'] = band_power / np.maximum(total, 1e-24)

        result = {}
        for stat, values in channel_stats.items():
            array = values.reshape(8, 8, 2)
            for modality, name in enumerate(('vibration', 'shock')):
                for side in (0, 1):
                    selected = array[:, side::2, modality]
                    prefix = f'{name}_side{side + 1}_{stat}'
                    result[f'{prefix}_mean'] = float(selected.mean())
                    result[f'{prefix}_max'] = float(selected.max())
                    for car in range(8):
                        result[f'car{car + 1}_{prefix}_mean'] = float(selected[car].mean())
                side_values = [float(array[:, side::2, modality].mean()) for side in (0, 1)]
                result[f'{name}_{stat}_side_difference'] = side_values[0] - side_values[1]
                denom = abs(side_values[0]) + abs(side_values[1]) + 1e-12
                result[f'{name}_{stat}_normalized_side_difference'] = (side_values[0] - side_values[1]) / denom

        if self.feature_names and len(self.feature_indices) > 0:
            feat_vec = np.array([result.get(name, 0.0) for name in self.feature_names])[self.feature_indices]
        else:
            global_keys = [k for k in result.keys() if not k.startswith('car')]
            feat_vec = np.array([result[k] for k in global_keys])

        return feat_vec.reshape(1, -1)

    def evaluate_chainage(self, current_kp: float, grounded_override: bool = False,
                          rng: np.random.Generator = None) -> dict:
        """
        Evaluates track corrugation at current chainage KP (km).
        If rail grinding was performed (grounded_override), severity drops to nominal.
        """
        if rng is None:
            rng = np.random.default_rng()

        if grounded_override:
            return {
                "kp_start": round(current_kp - 0.05, 3),
                "kp_end": round(current_kp + 0.05, 3),
                "depth_microns": 8.5,
                "wavelength_class": "SMOOTH_GROUND",
                "severity_score": 0.08,
                "maintenance_urgency": "NOMINAL",
                "grinding_priority_rank": 0
            }

        # Check proximity to known corrugation zones
        for zone in CORRUGATION_ZONES:
            if zone["kp_start"] - 0.05 <= current_kp <= zone["kp_end"] + 0.05:
                # Inside anomalous corrugated zone
                return {
                    "kp_start": zone["kp_start"],
                    "kp_end": zone["kp_end"],
                    "depth_microns": zone["depth_microns"],
                    "wavelength_class": zone["wavelength_class"],
                    "severity_score": zone["severity"],
                    "maintenance_urgency": zone["urgency"],
                    "grinding_priority_rank": zone["rank"]
                }

        # Nominal smooth rail
        noise_depth = float(rng.uniform(9.0, 14.5))
        return {
            "kp_start": round(current_kp - 0.05, 3),
            "kp_end": round(current_kp + 0.05, 3),
            "depth_microns": round(noise_depth, 1),
            "wavelength_class": "NOMINAL",
            "severity_score": round(noise_depth / 50.0, 2),
            "maintenance_urgency": "NONE",
            "grinding_priority_rank": 0
        }

    def predict_from_csv(self, file_source, file_name: str = "rail_data.csv") -> dict:
        """
        Processes an axle-box multi-channel shock/vibration CSV to detect rail corrugation.
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


        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            raise ValueError("Uploaded rail data must contain numeric sensor channels.")

        # Compute broadband RMS across vibration channels
        vib_cols = [c for c in num_cols if 'vibration' in str(c).lower() or 'shock' in str(c).lower()]
        if not vib_cols:
            vib_cols = num_cols

        channel_rms = [float(np.sqrt(np.mean(df[c].dropna().values**2))) for c in vib_cols]
        mean_rms = float(np.mean(channel_rms)) if channel_rms else 0.5
        max_rms = float(np.max(channel_rms)) if channel_rms else 0.5

        if len(num_cols) < 128:
            raise ValueError(
                f"Uploaded rail data '{file_name}' contains {len(num_cols)} numeric channels, but ML inference "
                f"strictly requires at least 128 axle-box vibration/shock channels. Heuristic fallback has been disabled."
            )

        if not self.models:
            raise RuntimeError("No trained Rail Corrugation models are loaded to perform ML inference.")

        features = self._extract_features_252(df)
        if features is None:
            raise ValueError(
                f"Feature extraction failed for '{file_name}'. Input data does not match the 128-channel format."
            )

        all_probs = []
        for m in self.models:
            try:
                p = m.predict_proba(features)[0]
                all_probs.append(p)
            except Exception as inf_err:
                print(f"[RailCorrugationModel] Model prediction error: {inf_err}")

        if not all_probs:
            raise RuntimeError(
                f"All {len(self.models)} corrugation model estimators failed to produce probability predictions for '{file_name}'."
            )

        mean_probs = np.mean(all_probs, axis=0)
        pred_idx = int(np.argmax(mean_probs))
        pred_label = self.class_labels[pred_idx] if pred_idx < len(self.class_labels) else "Normal"
        prob_normal = float(mean_probs[0])
        prob_side_i = float(mean_probs[1]) if len(mean_probs) > 1 else 0.0
        prob_side_ii = float(mean_probs[2]) if len(mean_probs) > 2 else 0.0

        if pred_label == 'Normal':
            status = "GOOD"
            verdict = "GOOD"
            severity_score = round(float(np.clip(1.0 - prob_normal, 0.04, 0.35)), 2)
            depth_microns = round(float(np.clip(10.0 + severity_score * 20.0, 8.0, 18.0)), 1)
            wavelength = "NOMINAL"
            urgency = "NONE"
            action = "NONE"
            rank = 0
            summary = (
                f"Track surface analysis for '{file_name}': smooth rail surface detected "
                f"(confidence {prob_normal*100:.1f}%). Mean vibration RMS is {mean_rms:.2f}g. "
                f"Estimated roughness depth {depth_microns} microns is well within safe operating limits."
            )
        else:
            is_severe = max(prob_side_i, prob_side_ii) > 0.65 or mean_rms > 1.5
            status = "ACTION_NEEDED" if is_severe else "WATCH"
            verdict = status
            severity_score = round(float(np.clip(max(prob_side_i, prob_side_ii), 0.45, 0.98)), 2)
            depth_microns = round(float(np.clip(28.0 + severity_score * 28.0, 25.0, 65.0)), 1)
            wavelength = "SHORT_PITCH" if is_severe else "LONG_PITCH"
            urgency = "IMMEDIATE_GRINDING_48H" if is_severe else "SCHEDULE_GRINDING_7D"
            action = "ACTION_GRIND_RAIL"
            rank = 1 if is_severe else 2
            summary = (
                f"Corrugation detected on {pred_label} for '{file_name}' (confidence {max(prob_side_i, prob_side_ii)*100:.1f}%). "
                f"Multi-channel vibration RMS reaches {mean_rms:.2f}g (peak {max_rms:.2f}g). "
                f"Estimated acoustic rail corrugation depth is {depth_microns} microns ({wavelength}). "
                f"Recommended: {urgency.replace('_', ' ').title()}."
            )

        return {
            "subsystem": "rail",
            "file_name": file_name,
            "status": status,
            "verdict": verdict,
            "predicted_class": pred_label,
            "probabilities": {
                "normal": round(prob_normal, 4),
                "side_i": round(prob_side_i, 4),
                "side_ii": round(prob_side_ii, 4)
            },
            "anomaly_score": severity_score,
            "depth_microns": depth_microns,
            "wavelength_class": wavelength,
            "maintenance_urgency": urgency,
            "grinding_priority_rank": rank,
            "recommended_action": action,
            "mean_channel_rms": round(mean_rms, 2),
            "conductor_summary": summary,
        }

    def predict_batch(self, file_list: list, batch_name: str = "rail_batch.zip") -> dict:
        """
        Evaluates a batch of Rail Corrugation vibration test files (e.g. from an uploaded ZIP).
        file_list: list of (file_name, file_bytes)
        """
        results = []
        for fname, fbytes in file_list:
            try:
                res = self.predict_from_csv(fbytes, fname)
                results.append(res)
            except Exception as e:
                print(f"[RailCorrugationModel] Error processing {fname} in batch: {e}")

        if not results:
            raise ValueError(f"Could not parse any valid Rail Corrugation CSV files from '{batch_name}'.")

        total_files = len(results)
        depths = [r["depth_microns"] for r in results]
        scores = [r["anomaly_score"] for r in results]
        rms_vals = [r["mean_channel_rms"] for r in results]

        mean_depth = round(float(np.mean(depths)), 1)
        max_depth = round(float(np.max(depths)), 1)
        worst_item = max(results, key=lambda r: r["anomaly_score"])
        worst_file = worst_item["file_name"]

        mean_rms = round(float(np.mean(rms_vals)), 2)
        max_rms = round(float(np.max(rms_vals)), 2)

        anomalous_files = [r for r in results if r["status"] != "GOOD"]
        anomaly_count = len(anomalous_files)

        if worst_item["anomaly_score"] > 0.65:
            verdict = "ACTION_NEEDED"
            action = "ACTION_GRIND_RAIL"
            conductor_summary = (
                f"Batch evaluation of {total_files} rail vibration runs in '{batch_name}': "
                f"Severe corrugation warning identified across {anomaly_count} test section(s). "
                f"Peak roughness detected in '{worst_file}' with estimated depth of {max_depth} microns "
                f"(severity {worst_item['anomaly_score']:.2f}, mean vibration RMS {worst_item['mean_channel_rms']:.2f}g). "
                f"Immediate rail grinding required within 48 hours to avert wheel-rail acoustic damage."
            )
        elif anomaly_count > 0:
            verdict = "WATCH"
            action = "ACTION_GRIND_RAIL"
            conductor_summary = (
                f"Batch evaluation of {total_files} rail vibration runs in '{batch_name}': "
                f"Moderate corrugation detected in {anomaly_count} of {total_files} sections. "
                f"Average roughness depth across batch is {mean_depth} microns (worst section '{worst_file}' at {max_depth} microns). "
                f"Rail grinding recommended within 7 days."
            )
        else:
            verdict = "GOOD"
            action = "NONE"
            conductor_summary = (
                f"Batch evaluation of {total_files} rail vibration runs in '{batch_name}': "
                f"All {total_files} track sections show smooth rail surface profiles "
                f"(batch mean depth {mean_depth} microns, mean vibration RMS {mean_rms:.2f}g). "
                f"Track acoustic and surface parameters are nominal."
            )

        batch_breakdown = [
            {
                "file": r["file_name"],
                "depth_microns": r["depth_microns"],
                "severity": r["anomaly_score"],
                "wavelength_class": r["wavelength_class"],
                "mean_channel_rms": r["mean_channel_rms"],
                "verdict": r["verdict"],
                "urgency": r["maintenance_urgency"]
            }
            for r in results
        ]

        return {
            "subsystem": "rail",
            "file_name": batch_name,
            "is_batch": True,
            "status": verdict,
            "verdict": verdict,
            "total_files": total_files,
            "anomaly_count": anomaly_count,
            "depth_microns": max_depth,
            "mean_depth_microns": mean_depth,
            "worst_file": worst_file,
            "mean_channel_rms": max_rms,
            "batch_mean_channel_rms": mean_rms,
            "wavelength_class": worst_item["wavelength_class"],
            "anomaly_score": worst_item["anomaly_score"],
            "maintenance_urgency": worst_item["maintenance_urgency"],
            "grinding_priority_rank": worst_item["grinding_priority_rank"],
            "recommended_action": action,
            "conductor_summary": conductor_summary,
            "batch_items": batch_breakdown,
        }


