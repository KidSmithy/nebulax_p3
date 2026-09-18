"""
rail_pipeline.py
=============================================================================
Clean Python Interface for Frontend / Web App / Streamlit / API.
Allows a non-technical user or frontend UI (Streamlit, Flask, FastAPI) to:
1. Load the trained production ensemble bundle with 1 line of code.
2. Upload / pass any raw CSV recording (or filepath).
3. Get back predictions, confidence probabilities, train speed, and health diagnostics.
=============================================================================

Usage Example:
--------------
from rail_pipeline import RailPredictor

predictor = RailPredictor()
result = predictor.predict("Test1.csv")
print(result)
# Output:
# {
#    'prediction': 'Normal',
#    'confidence': 0.962,
#    'speed_kmh': 48.2,
#    'status': 'Normal Healthy Track',
#    'probabilities': {'Normal': 0.962, 'Side I': 0.015, 'Side II': 0.023},
#    'diagnostic_summary': 'Model classified segment as Normal at 48.2 km/h with 96.2% confidence.'
# }
"""

import os
import io
import joblib
import numpy as np
import pandas as pd
from scipy import signal

# Default bundle location
DEFAULT_BUNDLE_PATH = os.path.join(os.path.dirname(__file__), "rail_model_bundle.joblib")

WHEEL_DIAMETER = 0.85
WHEEL_CIRC = np.pi * WHEEL_DIAMETER
FS = 10000

class RailPredictor:
    """Production Inference Engine for Rail Corrugation Monitoring."""
    
    def __init__(self, bundle_path: str = None):
        if bundle_path is None:
            bundle_path = DEFAULT_BUNDLE_PATH
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"Model bundle not found at {bundle_path}.")
            
        self.bundle = joblib.load(bundle_path)
        self.final_lgb = self.bundle.get('final_lgb', None)
        self.final_xt = self.bundle.get('final_xt', None)
        self.lgb_boosters = self.bundle.get('lgb_boosters', [])
        self.xt_boosters = self.bundle.get('xt_boosters', [])
        self.blend_weights = self.bundle.get('blend_weights', (0.6, 0.4))
        self.threshold_divisors = self.bundle.get('threshold_divisors', np.array([1.0, 1.0, 1.0]))
        self.feature_cols = self.bundle['feature_cols']
        self.inv_label_map = self.bundle.get('inv_label_map', {0: 'Normal', 1: 'Side I', 2: 'Side II'})
        self.label_map = self.bundle.get('label_map', {'Normal': 0, 'Side I': 1, 'Side II': 2})
        self.low_speed_th = self.bundle.get('speed_gate_kmh', 10.0)
        self.validation_metrics = self.bundle.get('validation_metrics', {})
        
    def _decode_speed(self, speed_series: np.ndarray) -> dict:
        diffs = np.abs(np.diff(speed_series))
        transitions = int(np.sum(diffs > 0))
        revs = transitions / 180.0
        v_ms = revs * WHEEL_CIRC
        v_kmh = v_ms * 3.6
        return {
            'transitions': transitions,
            'speed_ms': v_ms,
            'speed_kmh': v_kmh,
            'is_low_speed': 1 if v_kmh < 15.0 else 0,
            'is_zero_speed': 1 if transitions == 0 else 0
        }
        
    def extract_features(self, df: pd.DataFrame) -> dict:
        feats = {}
        speed_info = self._decode_speed(df['Rotating speed'].values)
        feats.update(speed_info)
        v_ms = speed_info['speed_ms']
        
        s1_vib_cols = [c for c in df.columns if 'Vibration' in c and any(f'position {p}' in c for p in [1, 3, 5, 7])]
        s2_vib_cols = [c for c in df.columns if 'Vibration' in c and any(f'position {p}' in c for p in [2, 4, 6, 8])]
        s1_shk_cols = [c for c in df.columns if 'Shock' in c and any(f'position {p}' in c for p in [1, 3, 5, 7])]
        s2_shk_cols = [c for c in df.columns if 'Shock' in c and any(f'position {p}' in c for p in [2, 4, 6, 8])]

        # Mean subtraction (DC offset removal)
        s1_v_mat = df[s1_vib_cols].values - np.mean(df[s1_vib_cols].values, axis=0)
        s2_v_mat = df[s2_vib_cols].values - np.mean(df[s2_vib_cols].values, axis=0)
        s1_s_mat = df[s1_shk_cols].values - np.mean(df[s1_shk_cols].values, axis=0)
        s2_s_mat = df[s2_shk_cols].values - np.mean(df[s2_shk_cols].values, axis=0)

        s1_v_rms = float(np.sqrt(np.mean(s1_v_mat**2)))
        s2_v_rms = float(np.sqrt(np.mean(s2_v_mat**2)))
        feats['s1_v_rms'] = s1_v_rms
        feats['s2_v_rms'] = s2_v_rms
        feats['v_rms_ratio'] = s1_v_rms / (s2_v_rms + 1e-8)
        feats['v_rms_diff'] = s1_v_rms - s2_v_rms

        s1_s_rms = float(np.sqrt(np.mean(s1_s_mat**2)))
        s2_s_rms = float(np.sqrt(np.mean(s2_s_mat**2)))
        feats['s1_s_rms'] = s1_s_rms
        feats['s2_s_rms'] = s2_s_rms
        feats['s_rms_ratio'] = s1_s_rms / (s2_s_rms + 1e-8)
        feats['s_rms_diff'] = s1_s_rms - s2_s_rms

        s1_v_ptp = float(np.mean(np.ptp(s1_v_mat, axis=0)))
        s2_v_ptp = float(np.mean(np.ptp(s2_v_mat, axis=0)))
        feats['v_ptp_ratio'] = s1_v_ptp / (s2_v_ptp + 1e-8)

        s1_kurt = float(np.mean(np.mean(s1_s_mat**4, axis=0) / (np.var(s1_s_mat, axis=0)**2 + 1e-8)))
        s2_kurt = float(np.mean(np.mean(s2_s_mat**4, axis=0) / (np.var(s2_s_mat, axis=0)**2 + 1e-8)))
        feats['s1_s_kurt'] = s1_kurt
        feats['s2_s_kurt'] = s2_kurt
        feats['s_kurt_ratio'] = s1_kurt / (s2_kurt + 1e-8)
        feats['s_kurt_diff'] = s1_kurt - s2_kurt

        car_v_diffs, car_v_ratios = [], []
        for car in range(1, 9):
            c_s1 = [f'Vibration of bearing in position {p} of car {car}' for p in [1, 3, 5, 7]]
            c_s2 = [f'Vibration of bearing in position {p} of car {car}' for p in [2, 4, 6, 8]]
            cv1 = np.sqrt(np.mean((df[c_s1].values - np.mean(df[c_s1].values, axis=0))**2))
            cv2 = np.sqrt(np.mean((df[c_s2].values - np.mean(df[c_s2].values, axis=0))**2))
            d = float(cv1 - cv2)
            r = float(cv1 / (cv2 + 1e-8))
            feats[f'car{car}_v_diff'] = d
            feats[f'car{car}_v_ratio'] = r
            car_v_diffs.append(d)
            car_v_ratios.append(r)

        sorted_ratios = np.sort(car_v_ratios)
        sorted_diffs = np.sort(car_v_diffs)
        feats['max_car_v_diff'] = float(sorted_diffs[-1])
        feats['min_car_v_diff'] = float(sorted_diffs[0])
        feats['max_car_v_ratio'] = float(sorted_ratios[-1])
        feats['min_car_v_ratio'] = float(sorted_ratios[0])
        feats['car_v_diff_range'] = feats['max_car_v_diff'] - feats['min_car_v_diff']

        feats['top1_car_ratio'] = float(sorted_ratios[-1])
        feats['top2_car_ratio'] = float(sorted_ratios[-2])
        feats['top3_car_ratio'] = float(sorted_ratios[-3])
        feats['mean_top2_car_ratio'] = float((sorted_ratios[-1] + sorted_ratios[-2]) / 2.0)
        feats['mean_top3_car_ratio'] = float((sorted_ratios[-1] + sorted_ratios[-2] + sorted_ratios[-3]) / 3.0)

        feats['top1_car_diff'] = float(sorted_diffs[-1])
        feats['top2_car_diff'] = float(sorted_diffs[-2])
        feats['mean_top2_car_diff'] = float((sorted_diffs[-1] + sorted_diffs[-2]) / 2.0)

        feats['min1_car_ratio'] = float(sorted_ratios[0])
        feats['min2_car_ratio'] = float(sorted_ratios[1])
        feats['mean_min2_car_ratio'] = float((sorted_ratios[0] + sorted_ratios[1]) / 2.0)
        feats['min1_car_diff'] = float(sorted_diffs[0])
        feats['min2_car_diff'] = float(sorted_diffs[1])
        feats['mean_min2_car_diff'] = float((sorted_diffs[0] + sorted_diffs[1]) / 2.0)

        feats['car_ratio_std'] = float(np.std(car_v_ratios))
        feats['car_ratio_range'] = float(sorted_ratios[-1] - sorted_ratios[0])

        mean_s1_sig = np.mean(s1_v_mat, axis=1)
        mean_s2_sig = np.mean(s2_v_mat, axis=1)
        f, pxx1 = signal.welch(mean_s1_sig, fs=FS, nperseg=1024)
        f, pxx2 = signal.welch(mean_s2_sig, fs=FS, nperseg=1024)

        sum1 = np.sum(pxx1) + 1e-12
        sum2 = np.sum(pxx2) + 1e-12
        cent1 = np.sum(f * pxx1) / sum1
        cent2 = np.sum(f * pxx2) / sum2
        pn1 = pxx1 / sum1; pn1 = pn1[pn1 > 0]
        pn2 = pxx2 / sum2; pn2 = pn2[pn2 > 0]
        ent1 = -np.sum(pn1 * np.log2(pn1))
        ent2 = -np.sum(pn2 * np.log2(pn2))
        feats['spectral_centroid_diff'] = float(cent1 - cent2)
        feats['spectral_entropy_diff'] = float(ent1 - ent2)

        fixed_bands = [
            ('band_0_100', 0, 100),
            ('band_100_250', 100, 250),
            ('band_250_500', 250, 500),
            ('band_500_1000', 500, 1000),
            ('band_1000_2000', 1000, 2000),
            ('band_2000_4000', 2000, 4000),
        ]
        for b_name, low, high in fixed_bands:
            idx = (f >= low) & (f < high)
            p1 = float(np.sum(pxx1[idx]))
            p2 = float(np.sum(pxx2[idx]))
            feats[f'{b_name}_s1'] = p1
            feats[f'{b_name}_s2'] = p2
            feats[f'{b_name}_ratio'] = float(p1 / (p2 + 1e-8))
            feats[f'{b_name}_diff'] = float(p1 - p2)

        if v_ms > 5.0:
            peak_idx1 = np.argmax(pxx1[f >= 100]) + int(100 / (FS / 1024))
            peak_f1 = f[peak_idx1]
            feats['dominant_wavelength_s1'] = float(v_ms / (peak_f1 + 1e-6))
        else:
            feats['dominant_wavelength_s1'] = 0.0

        if v_ms > 3.0:
            w_bins = [
                ('wave_2_4cm', 0.02, 0.04),
                ('wave_4_7cm', 0.04, 0.07),
                ('wave_7_12cm', 0.07, 0.12),
                ('wave_12_20cm', 0.12, 0.20),
                ('wave_20_30cm', 0.20, 0.30)
            ]
            for wname, lmin, lmax in w_bins:
                idx_w = (f >= v_ms / lmax) & (f < v_ms / lmin)
                pw1 = float(np.sum(pxx1[idx_w]))
                pw2 = float(np.sum(pxx2[idx_w]))
                feats[f'{wname}_ratio'] = float(pw1 / (pw2 + 1e-8))
                feats[f'{wname}_diff'] = float(pw1 - pw2)
        else:
            for wname in ['wave_2_4cm', 'wave_4_7cm', 'wave_7_12cm', 'wave_12_20cm', 'wave_20_30cm']:
                feats[f'{wname}_ratio'] = 1.0
                feats[f'{wname}_diff'] = 0.0

        # Physically Grounded Interaction & Speed-Normalized Features
        v_norm_factor = max(speed_info['speed_kmh'] / 50.0, 0.2)**2
        feats['v_rms_diff_speed_norm'] = feats['v_rms_diff'] / v_norm_factor
        feats['s1_v_rms_speed_norm'] = feats['s1_v_rms'] / v_norm_factor
        feats['tail_cars_v_ratio'] = (feats['car6_v_ratio'] + feats['car7_v_ratio'] + feats['car8_v_ratio']) / 3.0
        feats['lead_cars_v_ratio'] = (feats['car1_v_ratio'] + feats['car2_v_ratio'] + feats['car3_v_ratio']) / 3.0
        feats['tail_vs_lead_diff'] = feats['tail_cars_v_ratio'] - feats['lead_cars_v_ratio']
        feats['wave_4_20cm_ratio'] = (feats['wave_4_7cm_ratio'] + feats['wave_7_12cm_ratio'] + feats['wave_12_20cm_ratio']) / 3.0
        feats['asymmetry_x_corrugation'] = feats['v_rms_diff'] * feats['wave_7_12cm_ratio']

        return feats
        
    def predict(self, df_or_filepath) -> dict:
        """
        Accepts a filepath (str), BytesIO buffer, or pandas DataFrame.
        Returns a rich prediction dictionary for frontend consumption.
        """
        if isinstance(df_or_filepath, str):
            df = pd.read_csv(df_or_filepath)
        elif isinstance(df_or_filepath, (io.BytesIO, io.StringIO)):
            df = pd.read_csv(df_or_filepath)
        elif isinstance(df_or_filepath, pd.DataFrame):
            df = df_or_filepath
        else:
            raise ValueError("Input must be a filepath string, BytesIO buffer, or pandas DataFrame.")
            
        feats = self.extract_features(df)
        speed_kmh = round(feats['speed_kmh'], 2)
        
        status_desc = {
            'Normal': 'Normal Healthy Track',
            'Side I': 'Abnormal Rail Corrugation Detected on Side I',
            'Side II': 'Abnormal Rail Corrugation Detected on Side II'
        }
        
        # 1. Deterministic Low-Speed Gating (< 10 km/h)
        if speed_kmh < self.low_speed_th:
            return {
                'prediction': 'Normal',
                'confidence': 1.0,
                'speed_kmh': speed_kmh,
                'status': 'Normal (Low / Stationary Speed)',
                'probabilities': {'Normal': 1.0, 'Side I': 0.0, 'Side II': 0.0},
                'diagnostic_summary': f"Train speed ({speed_kmh} km/h) is below dynamic excitation threshold (<{self.low_speed_th} km/h). Track condition is classified as Normal.",
                'asymmetry_ratio': round(feats['v_rms_ratio'], 3),
                'top1_car_ratio': round(feats['top1_car_ratio'], 3),
                'top2_car_ratio': round(feats['top2_car_ratio'], 3),
                'dominant_wavelength_s1_cm': round(feats['dominant_wavelength_s1'] * 100, 2)
            }
            
        # 2. Ensemble Inference
        x_vec = np.array([[feats[col] for col in self.feature_cols]])
        
        w_lgb, w_xt = self.blend_weights
        
        if self.final_lgb is not None and self.final_xt is not None:
            p_lgb = self.final_lgb.predict_proba(x_vec)[0]
            p_xt = self.final_xt.predict_proba(x_vec)[0]
            raw_probs = w_lgb * p_lgb + w_xt * p_xt
        elif self.lgb_boosters and self.xt_boosters:
            p_lgb_folds = [b.predict(x_vec)[0] for b in self.lgb_boosters]
            p_lgb = np.mean(p_lgb_folds, axis=0)
            p_xt_folds = [b.predict(x_vec)[0] for b in self.xt_boosters]
            p_xt = np.mean(p_xt_folds, axis=0)
            raw_probs = w_lgb * p_lgb + w_xt * p_xt
        else:
            raise RuntimeError("No valid model found in loaded bundle.")
        
        # Calibrated decision threshold divisors
        adj_scores = raw_probs / self.threshold_divisors
        pred_idx = int(np.argmax(adj_scores))
        pred_label = self.inv_label_map[pred_idx]
        confidence = float(raw_probs[pred_idx])
        
        prob_dict = {
            'Normal': round(float(raw_probs[self.label_map['Normal']]), 4),
            'Side I': round(float(raw_probs[self.label_map['Side I']]), 4),
            'Side II': round(float(raw_probs[self.label_map['Side II']]), 4),
        }
        
        return {
            'prediction': pred_label,
            'confidence': round(confidence, 4),
            'speed_kmh': speed_kmh,
            'status': status_desc.get(pred_label, pred_label),
            'probabilities': prob_dict,
            'diagnostic_summary': f"Model classified track recording as {pred_label} at {speed_kmh} km/h with {round(confidence*100, 1)}% calibrated confidence.",
            'asymmetry_ratio': round(feats['v_rms_ratio'], 3),
            'top1_car_ratio': round(feats['top1_car_ratio'], 3),
            'top2_car_ratio': round(feats['top2_car_ratio'], 3),
            'dominant_wavelength_s1_cm': round(feats['dominant_wavelength_s1'] * 100, 2)
        }

if __name__ == "__main__":
    predictor = RailPredictor()
    test_sample = r"c:\Users\Rald999\Documents\GitHub\nebulax_p3\PS3\02_Datasets\Rail_Corrugation\Test\Test1.csv"
    if os.path.exists(test_sample):
        res = predictor.predict(test_sample)
        print("Self-Test Output for Test1.csv:")
        for k, v in res.items():
            print(f"  {k}: {v}")
