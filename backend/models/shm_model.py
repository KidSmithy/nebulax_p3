"""
backend/models/shm_model.py
===============================================================================
Structural Health Monitoring (SHM) Subsystem Model:
- Evaluates bogie and axle-box acceleration (vibration RMS g)
- Computes FFT spectrum, Welch PSD, Rainflow fatigue damage index
- Predicts bearing defect probability and critical weld node stress
===============================================================================
"""

import os
import io
import pickle
import numpy as np
import pandas as pd
from scipy.signal import welch
from backend.core.config import SHM_DIR, PS3_DIR, MODEL_DATA_DIR

MODEL_PATHS = [
    MODEL_DATA_DIR / "shm_model.pkl",
    SHM_DIR / "shm_model.pkl",
    PS3_DIR / "models" / "shm_model.pkl"
]

class SHMSubsystemModel:
    def __init__(self):
        self.payload = None
        self.scaler = None
        self.models = []
        self.weights = []
        self.features = [
            'p2p', 'abs_peak', 'min', 'valley_min', 'psd_total_energy', 'mad',
            'rf_damage_proxy_m3.5', 'psd_energy_0_10Hz', 'rf_damage_proxy_m4.0',
            'rf_damage_proxy_m3.3', 'iqr', 'rf_damage_proxy_m4.5'
        ]
        self.clip_range = (0.025, 0.950)
        self._load_model()

    def _load_model(self):
        for p in MODEL_PATHS:
            if p.exists():
                try:
                    with open(p, "rb") as f:
                        self.payload = pickle.load(f)
                    self.scaler = self.payload.get('scaler')
                    self.models = self.payload.get('models', [])
                    self.weights = self.payload.get('weights', [])
                    if 'features' in self.payload:
                        self.features = self.payload['features']
                    if 'clip_range' in self.payload:
                        self.clip_range = self.payload['clip_range']
                    print(f"[SHMModel] Loaded trained model payload from {p} with {len(self.models)} ensemble regressors")
                    return
                except Exception as e:
                    print(f"[SHMModel] Warning loading model from {p}: {e}")

    def evaluate_vibration(
        self,
        raw_vib_samples: np.ndarray = None,
        speed_kmh: float = 68.4,
        corrugation_severity: float = 0.0,
        bearing_wear: float = 0.0,
        rng: np.random.Generator = None,
    ) -> dict:
        """
        Evaluates bogie dynamic structural stress and FFT harmonics.
        Couples with track corrugation severity (physical cross-correlation).

        bearing_wear (0..1) is the axle-box bearing's own mechanical condition.
        It is deliberately INDEPENDENT of track roughness: a spalled inner race
        radiates energy at a discrete defect frequency whether or not the rail
        is corrugated. Servicing the bearing therefore reduces bearing risk
        without needing the track to be ground.

        rng: pass an explicit generator to make the frame reproducible. The
        harmonizer relies on this to compare an intervened frame against its
        own baseline without sampling noise polluting the delta.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Cross-correlation: rail corrugation excites axle-box vibration
        base_rms = 1.2 + (speed_kmh / 80.0) * 0.8
        corrugation_boost = corrugation_severity * 2.8
        # A worn bearing adds its own broadband contribution on top.
        bearing_boost = bearing_wear * 0.9
        vibration_rms = float(np.clip(
            base_rms + corrugation_boost + bearing_boost + rng.normal(0, 0.1), 0.8, 6.5
        ))

        # Ball-pass inner-race defect frequency scales with shaft speed.
        bearing_defect_freq = round(float(180.0 + speed_kmh * 0.35), 1)

        # Peak frequency shifts based on speed, corrugation and bearing state.
        if corrugation_severity > 0.4:
            peak_freq = 142.5
        elif bearing_wear > 0.35:
            peak_freq = bearing_defect_freq
        else:
            peak_freq = 95.0 + speed_kmh * 0.4

        # Fatigue accumulates from the structural load, i.e. mostly track input.
        fatigue_index = float(np.clip((vibration_rms / 4.0) ** 1.8 * 0.65, 0.08, 0.98))

        # Bearing risk is driven by its own condition, with a smaller
        # aggravating contribution from the shock loading the track imposes.
        bearing_defect_prob = float(np.clip(
            bearing_wear * 0.85 + corrugation_severity * 0.15, 0.03, 0.95
        ))

        # Anomaly score calibrated against the published thresholds: 2.5 g is
        # the top of the healthy band and 3.5 g is the concern threshold, so a
        # nominally-running bogie must not score as half-anomalous.
        anomaly_score = float(np.clip((vibration_rms - 1.5) / 3.5, 0.05, 0.99))
        critical_node = "BOLSTER_RIB_L2" if anomaly_score > 0.60 else "SIDE_BEAM_R1"

        # Generate representative FFT spectrum (fundamental + harmonics +
        # track excitation + discrete bearing defect tone)
        freq_grid = np.array([12.5, 25.0, 48.0, 75.0, 95.0, 120.0, 142.5, 180.0, 240.0, 300.0, 380.0, 450.0])
        amps = []
        for f in freq_grid:
            # Base background noise
            a = 0.15 + float(rng.uniform(0.01, 0.08))
            # Excitation near peak frequency
            dist = abs(f - peak_freq)
            if dist < 25.0:
                a += (vibration_rms * 0.75) * np.exp(-0.5 * (dist / 12.0) ** 2)
            # Discrete bearing defect tone
            bdist = abs(f - bearing_defect_freq)
            if bearing_wear > 0.05 and bdist < 30.0:
                a += (bearing_wear * 2.4) * np.exp(-0.5 * (bdist / 14.0) ** 2)
            # Structural resonance at 25 Hz and 300 Hz
            if abs(f - 25.0) < 5.0:
                a += 0.35 * (speed_kmh / 70.0)
            if abs(f - 300.0) < 15.0:
                a += 0.18 * vibration_rms
            amps.append(round(float(a), 2))

        fft_spectrum = [
            {"freq_hz": float(f), "amp": float(a)}
            for f, a in zip(freq_grid, amps)
        ]

        return {
            "bogie_id": "BOGIE_FRONT",
            "vibration_rms_g": round(vibration_rms, 2),
            "peak_frequency_hz": round(float(peak_freq), 1),
            "bearing_defect_prob": round(bearing_defect_prob, 2),
            "bearing_defect_freq_hz": bearing_defect_freq,
            "anomaly_score": round(anomaly_score, 2),
            "critical_weld_node": critical_node,
            "fft_spectrum": fft_spectrum
        }

    @staticmethod
    def _get_turning_points(series: np.ndarray) -> np.ndarray:
        diff1 = np.diff(series)
        turning = np.where(diff1[:-1] * diff1[1:] <= 0)[0] + 1
        return series[np.concatenate(([0], turning, [len(series) - 1]))]

    @staticmethod
    def _rainflow_fast(extrema: np.ndarray):
        stack = []
        ranges = []
        counts = []
        for x in extrema:
            stack.append(x)
            while len(stack) >= 3:
                s0, s1, s2 = stack[-3], stack[-2], stack[-1]
                r1 = abs(s1 - s0)
                r2 = abs(s2 - s1)
                if r2 >= r1:
                    if len(stack) == 3:
                        ranges.append(r1)
                        counts.append(0.5)
                        stack.pop(1)
                    else:
                        ranges.append(r1)
                        counts.append(1.0)
                        stack.pop(-2)
                        stack.pop(-2)
                else:
                    break
        while len(stack) > 1:
            ranges.append(abs(stack[-1] - stack[-2]))
            counts.append(0.5)
            stack.pop()
        return np.array(ranges, dtype=np.float32), np.array(counts, dtype=np.float32)

    def predict_from_csv(self, file_source, file_name: str = "shm_data.csv") -> dict:
        """
        Runs ASTM E1049-85 Rainflow cycle counting, Welch PSD, and statistical
        moment extraction on an uploaded SHM accelerometer/strain sensor CSV,
        then evaluates the multi-model ensemble (BayesianRidge, ElasticNet, SVR, ExtraTrees).
        """
        if isinstance(file_source, (bytes, bytearray)):
            try:
                df = pd.read_csv(io.BytesIO(file_source), header=None, dtype=np.float32)
            except Exception:
                df = pd.read_csv(io.BytesIO(file_source))
        elif isinstance(file_source, str) and "\n" in file_source:
            df = pd.read_csv(io.StringIO(file_source), header=None, dtype=np.float32)
        else:
            try:
                df = pd.read_csv(file_source, header=None, dtype=np.float32)
            except Exception:
                df = pd.read_csv(file_source)

        # Extract numeric sensor signal array
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            raise ValueError("Uploaded SHM CSV must contain numeric accelerometer/strain sensor values.")
        
        # Use primary acceleration / strain sensor column
        sig = df[num_cols[0]].dropna().values.astype(np.float32).flatten()
        if len(sig) < 64:
            raise ValueError(f"Uploaded SHM data must have at least 64 samples (received {len(sig)}).")

        # Subsample if extremely large to maintain responsiveness (< 250,000 samples)
        if len(sig) > 250000:
            step = len(sig) // 250000 + 1
            sig_eval = sig[::step]
        else:
            sig_eval = sig

        mean_val = float(np.mean(sig_eval))
        min_val = float(np.min(sig_eval))
        max_val = float(np.max(sig_eval))
        p2p_val = float(max_val - min_val)
        abs_peak = float(np.max(np.abs(sig_eval)))
        mad_val = float(np.mean(np.abs(sig_eval - mean_val)))
        rms_val = float(np.sqrt(np.mean(sig_eval**2)))

        pct_vals = np.percentile(sig_eval, [25, 75])
        iqr_val = float(pct_vals[1] - pct_vals[0])

        # Rainflow cycle analysis
        extrema = self._get_turning_points(sig_eval)
        rf_ranges, rf_counts = self._rainflow_fast(extrema)
        amps = rf_ranges / 2.0

        rf_proxy_33 = float(np.sum(rf_counts * (amps ** 3.3)))
        rf_proxy_35 = float(np.sum(rf_counts * (amps ** 3.5)))
        rf_proxy_40 = float(np.sum(rf_counts * (amps ** 4.0)))
        rf_proxy_45 = float(np.sum(rf_counts * (amps ** 4.5)))

        # Welch PSD spectral distribution
        freqs, psd = welch(sig_eval, fs=256.0, nperseg=min(2048, len(sig_eval)))
        psd_total = float(np.sum(psd))
        psd_0_10 = float(np.sum(psd[np.where((freqs >= 0) & (freqs < 10))[0]]))
        peak_idx = int(np.argmax(psd))
        dominant_freq = round(float(freqs[peak_idx]), 1) if peak_idx < len(freqs) else 95.0

        feat_dict = {
            'p2p': p2p_val,
            'abs_peak': abs_peak,
            'min': min_val,
            'valley_min': min_val,
            'psd_total_energy': psd_total,
            'mad': mad_val,
            'rf_damage_proxy_m3.5': rf_proxy_35,
            'psd_energy_0_10Hz': psd_0_10,
            'rf_damage_proxy_m4.0': rf_proxy_40,
            'rf_damage_proxy_m3.3': rf_proxy_33,
            'iqr': iqr_val,
            'rf_damage_proxy_m4.5': rf_proxy_45
        }

        # Model ensemble regression
        if self.scaler is not None and len(self.models) > 0:
            x_raw = np.array([[feat_dict.get(f, 0.0) for f in self.features]])
            x_scaled = self.scaler.transform(x_raw)
            preds = [float(np.exp(m.predict(x_scaled)[0])) for m in self.models]
            if len(self.weights) == len(preds):
                weighted_pred = sum(p * w for p, w in zip(preds, self.weights))
            else:
                weighted_pred = float(np.mean(preds))
            clip_min, clip_max = self.clip_range
            fatigue_damage_index = round(float(np.clip(weighted_pred, clip_min, clip_max)), 4)
        else:
            # Physics-based baseline approximation
            fatigue_damage_index = round(float(np.clip((rms_val / 4.0) ** 1.8 * 0.65, 0.05, 0.95)), 4)

        # Bearing defect probability estimation
        # Characteristic ball-pass impact ratio
        bearing_wear_est = float(np.clip((abs_peak / (rms_val + 1e-4) - 2.5) * 0.25 + (p2p_val / 15.0) * 0.2, 0.04, 0.95))
        bearing_defect_prob = round(bearing_wear_est, 2)

        # Overall anomaly score calibrated to published thresholds:
        # RMS healthy <= 2.5 g, warn <= 3.5 g; Fatigue healthy <= 0.40, warn <= 0.70
        rms_severity = max(0.0, (rms_val - 2.0) / 2.5)
        fatigue_severity = max(0.0, (fatigue_damage_index - 0.20) / 0.60)
        anomaly_score = round(float(np.clip(0.5 * rms_severity + 0.5 * fatigue_severity, 0.05, 0.98)), 2)

        if anomaly_score < 0.40 and fatigue_damage_index < 0.40 and rms_val < 2.5:
            status = "GOOD"
            action = "NONE"
            critical_node = "SIDE_BEAM_R1"
            conductor_summary = (
                f"Evaluated {len(sig):,} sensor points in '{file_name}'. "
                f"Vibration RMS is nominal at {rms_val:.2f}g (peak {abs_peak:.2f}g). "
                f"Rainflow fatigue damage proxy index is {fatigue_damage_index:.3f} "
                f"and bearing defect probability is low ({bearing_defect_prob*100:.0f}%). "
                f"Bogie structural integrity is sound."
            )
        elif anomaly_score < 0.65 and fatigue_damage_index < 0.70:
            status = "WATCH"
            action = "ACTION_INSPECT_BEARING" if bearing_defect_prob > 0.35 else "NONE"
            critical_node = "BOLSTER_RIB_L2"
            conductor_summary = (
                f"Caution on '{file_name}': sensor analysis indicates moderate vibration "
                f"({rms_val:.2f}g RMS, peak {abs_peak:.2f}g) and accumulated fatigue index {fatigue_damage_index:.3f}. "
                f"Bearing defect probability is elevated at {bearing_defect_prob*100:.0f}%. "
                f"Recommend scheduling axle-box bearing inspection during next depot turn."
            )
        else:
            status = "ACTION_NEEDED"
            action = "ACTION_INSPECT_BEARING"
            critical_node = "BOLSTER_RIB_L2"
            conductor_summary = (
                f"High-severity warning for '{file_name}': extreme structural acceleration detected "
                f"({rms_val:.2f}g RMS, peak {abs_peak:.2f}g). Rainflow fatigue index reaches {fatigue_damage_index:.3f} "
                f"with bearing defect probability at {bearing_defect_prob*100:.0f}%. "
                f"Critical stress localized at {critical_node}. Immediate bogie maintenance and bearing replacement advised."
            )

        # Construct sampled FFT snippet for display
        spectrum_pts = []
        step_f = max(1, len(freqs) // 15)
        for idx in range(0, min(len(freqs), step_f * 15), step_f):
            spectrum_pts.append({
                "freq_hz": round(float(freqs[idx]), 1),
                "amp": round(float(psd[idx]), 3)
            })

        return {
            "subsystem": "shm",
            "file_name": file_name,
            "status": status,
            "verdict": status,
            "sample_count": len(sig),
            "vibration_rms_g": round(rms_val, 2),
            "abs_peak_g": round(abs_peak, 2),
            "peak2peak_g": round(p2p_val, 2),
            "dominant_freq_hz": dominant_freq,
            "fatigue_damage_index": fatigue_damage_index,
            "bearing_defect_prob": bearing_defect_prob,
            "anomaly_score": anomaly_score,
            "critical_weld_node": critical_node,
            "recommended_action": action,
            "conductor_summary": conductor_summary,
            "fft_spectrum": spectrum_pts,
        }

    def predict_batch(self, file_list: list, batch_name: str = "shm_batch.zip") -> dict:
        """
        Evaluates a batch of SHM sensor files (e.g. from an uploaded ZIP).
        file_list: list of (file_name, file_bytes)
        """
        results = []
        for fname, fbytes in file_list:
            try:
                res = self.predict_from_csv(fbytes, fname)
                results.append(res)
            except Exception as e:
                print(f"[SHMModel] Error processing {fname} in batch: {e}")

        if not results:
            raise ValueError(f"Could not parse any valid SHM sensor CSV files from '{batch_name}'.")

        total_files = len(results)
        damage_indices = [r["fatigue_damage_index"] for r in results]
        rms_values = [r["vibration_rms_g"] for r in results]
        bearing_probs = [r["bearing_defect_prob"] for r in results]

        mean_damage = round(float(np.mean(damage_indices)), 4)
        max_damage = round(float(np.max(damage_indices)), 4)
        worst_damage_item = max(results, key=lambda r: r["fatigue_damage_index"])
        worst_file = worst_damage_item["file_name"]

        mean_rms = round(float(np.mean(rms_values)), 2)
        max_rms = round(float(np.max(rms_values)), 2)
        max_bearing_prob = round(float(np.max(bearing_probs)), 2)

        anomalous_files = [r for r in results if r["status"] != "GOOD"]
        anomaly_count = len(anomalous_files)

        if max_damage > 0.40 or max_bearing_prob > 0.45 or max_rms > 3.5:
            verdict = "ACTION_NEEDED"
            action = "ACTION_INSPECT_BEARING"
            conductor_summary = (
                f"Batch evaluation of {total_files} SHM files in '{batch_name}': "
                f"Severe structural risk detected in {anomaly_count} run(s). "
                f"Worst-case test run is '{worst_file}' with fatigue damage index {max_damage:.3f} "
                f"and peak bearing defect probability {max_bearing_prob*100:.0f}%. "
                f"Maximum vibration reached {max_rms:.2f}g. "
                f"Immediate axle-box bearing inspection and structural weld testing recommended."
            )
        elif anomaly_count > 0:
            verdict = "WATCH"
            action = "ACTION_INSPECT_BEARING"
            conductor_summary = (
                f"Batch evaluation of {total_files} SHM files in '{batch_name}': "
                f"Elevated stress observed in {anomaly_count} of {total_files} runs. "
                f"Average fatigue damage index across batch is {mean_damage:.3f} (max {max_damage:.3f} in '{worst_file}'). "
                f"Bearing inspection advised at upcoming depot service."
            )
        else:
            verdict = "GOOD"
            action = "NONE"
            conductor_summary = (
                f"Batch evaluation of {total_files} SHM files in '{batch_name}': "
                f"All {total_files} sensor runs are structurally nominal (mean RMS {mean_rms:.2f}g, "
                f"average fatigue damage index {mean_damage:.3f}). "
                f"Bogie structural health and axle bearings are in good operational condition."
            )

        batch_breakdown = [
            {
                "file": r["file_name"],
                "damage_index": r["fatigue_damage_index"],
                "rms_g": r["vibration_rms_g"],
                "bearing_risk_pct": round(r["bearing_defect_prob"] * 100),
                "verdict": r["verdict"],
                "critical_node": r["critical_weld_node"]
            }
            for r in results
        ]

        return {
            "subsystem": "shm",
            "file_name": batch_name,
            "is_batch": True,
            "status": verdict,
            "verdict": verdict,
            "total_files": total_files,
            "anomaly_count": anomaly_count,
            "fatigue_damage_index": max_damage,
            "mean_fatigue_damage_index": mean_damage,
            "worst_file": worst_file,
            "vibration_rms_g": max_rms,
            "mean_vibration_rms_g": mean_rms,
            "bearing_defect_prob": max_bearing_prob,
            "anomaly_score": worst_damage_item["anomaly_score"],
            "critical_weld_node": worst_damage_item["critical_weld_node"],
            "recommended_action": action,
            "conductor_summary": conductor_summary,
            "batch_items": batch_breakdown,
        }


