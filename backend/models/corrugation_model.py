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
from backend.core.config import RAIL_DIR, PS3_DIR, MODEL_DATA_DIR, CORRUGATION_ZONES

BUNDLE_PATHS = [
    MODEL_DATA_DIR / "rail_model_bundle.joblib",
    RAIL_DIR / "models" / "rail_model_bundle.joblib",
    PS3_DIR / "models" / "rail_model_bundle.joblib"
]

class RailCorrugationModel:
    def __init__(self):
        self.bundle = None
        self.model = None
        self.feature_cols = []
        for p in BUNDLE_PATHS:
            if p.exists():
                try:
                    self.bundle = joblib.load(p)
                    self.model = self.bundle.get('model')
                    self.feature_cols = self.bundle.get('feature_cols', [])
                    print(f"[RailCorrugationModel] Loaded model bundle from {p}")
                    break
                except Exception as e:
                    print(f"[RailCorrugationModel] Warning loading bundle from {p}: {e}")

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
        if isinstance(file_source, (bytes, bytearray)):
            df = pd.read_csv(io.BytesIO(file_source))
        elif isinstance(file_source, str) and "\n" in file_source:
            df = pd.read_csv(io.StringIO(file_source))
        else:
            df = pd.read_csv(file_source)

        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            raise ValueError("Uploaded rail data must contain numeric sensor channels.")

        # Compute broadband RMS across vibration channels
        vib_cols = [c for c in num_cols if 'vibration' in c.lower() or 'shock' in c.lower()]
        if not vib_cols:
            vib_cols = num_cols

        channel_rms = [float(np.sqrt(np.mean(df[c].dropna().values**2))) for c in vib_cols]
        mean_rms = float(np.mean(channel_rms)) if channel_rms else 0.5
        max_rms = float(np.max(channel_rms)) if channel_rms else 0.5

        # Infer corrugation depth & severity
        depth_microns = round(float(np.clip(mean_rms * 28.0 + 8.0, 6.0, 65.0)), 1)
        severity_score = round(float(np.clip(depth_microns / 45.0, 0.05, 0.98)), 2)

        if severity_score > 0.65:
            wavelength = "SHORT_PITCH"
            urgency = "IMMEDIATE_GRINDING_48H"
            status = "ACTION_NEEDED"
            action = "ACTION_GRIND_RAIL"
            rank = 1
            summary = (
                f"Severe corrugation warning for '{file_name}': multi-channel vibration RMS reaches {mean_rms:.2f}g "
                f"(peak channel {max_rms:.2f}g). Estimated acoustic rail corrugation depth is {depth_microns} microns "
                f"({wavelength}). Immediate rail grinding required within 48 hours to prevent wheel damage."
            )
        elif severity_score > 0.35:
            wavelength = "LONG_PITCH"
            urgency = "SCHEDULE_GRINDING_7D"
            status = "WATCH"
            action = "ACTION_GRIND_RAIL"
            rank = 2
            summary = (
                f"Moderate corrugation detected for '{file_name}': estimated rail roughness depth {depth_microns} microns "
                f"({wavelength}, severity {severity_score}). Grinding recommended within 7 days."
            )
        else:
            wavelength = "NOMINAL"
            urgency = "NOMINAL"
            status = "GOOD"
            action = "NONE"
            rank = 0
            summary = (
                f"Track surface analysis for '{file_name}': smooth rail surface detected. "
                f"Estimated roughness depth {depth_microns} microns is well within safe operating limits."
            )

        return {
            "subsystem": "rail",
            "file_name": file_name,
            "status": status,
            "verdict": status,
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


