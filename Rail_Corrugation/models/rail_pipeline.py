"""
rail_pipeline.py
=============================================================================
Clean Python Interface for Frontend / Web App / API.
Allows a non-technical user or frontend UI (Streamlit, Flask, FastAPI) to:
1. Load the trained baseline model bundle with 1 line of code.
2. Upload / pass any raw CSV recording (or filepath).
3. Get back predictions, confidence probabilities, train speed, and health diagnostics.
=============================================================================

Usage Example:
--------------
from rail_pipeline import RailPredictor

predictor = RailPredictor()
result = predictor.predict_file("Test1.csv")
print(result)
# Output:
# {
#    'prediction': 'Normal',
#    'confidence': 0.942,
#    'speed_kmh': 48.2,
#    'status': 'Healthy',
#    'probabilities': {'Normal': 0.942, 'Side I': 0.021, 'Side II': 0.037},
#    'reason': 'Vibration levels balanced across rails'
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

class RailPredictor:
    """Production Inference Engine for Rail Corrugation Monitoring."""
    
    def __init__(self, bundle_path: str = None):
        if bundle_path is None:
            bundle_path = DEFAULT_BUNDLE_PATH
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"Model bundle not found at {bundle_path}. Run train_and_export_baseline.py first.")
            
        self.bundle = joblib.load(bundle_path)
        self.model = self.bundle['model']
        self.feature_cols = self.bundle['feature_cols']
        self.inv_label_map = self.bundle['inv_label_map']
        self.label_map = self.bundle['label_map']
        self.low_speed_th = self.bundle.get('speed_gating_threshold_kmh', 10.0)
        
    def _decode_speed(self, speed_series: np.ndarray) -> dict:
        diffs = np.abs(np.diff(speed_series))
        transitions = int(np.sum(diffs > 0))
        revs = transitions / 180.0
        v_ms = revs * (np.pi * 0.85)
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
        
        s1_vib_cols = [c for c in df.columns if "Vibration" in c and any(f"position {p}" in c for p in [1, 3, 5, 7])]
        s2_vib_cols = [c for c in df.columns if "Vibration" in c and any(f"position {p}" in c for p in [2, 4, 6, 8])]
        s1_shk_cols = [c for c in df.columns if "Shock" in c and any(f"position {p}" in c for p in [1, 3, 5, 7])]
        s2_shk_cols = [c for c in df.columns if "Shock" in c and any(f"position {p}" in c for p in [2, 4, 6, 8])]
        
        # DC offset removal
        s1_v_mat = df[s1_vib_cols].values
        s2_v_mat = df[s2_vib_cols].values
        s1_s_mat = df[s1_shk_cols].values
        s2_s_mat = df[s2_shk_cols].values
        
        s1_v_mat_dyn = s1_v_mat - np.mean(s1_v_mat, axis=0)
        s2_v_mat_dyn = s2_v_mat - np.mean(s2_v_mat, axis=0)
        s1_s_mat_dyn = s1_s_mat - np.mean(s1_s_mat, axis=0)
        s2_s_mat_dyn = s2_s_mat - np.mean(s2_s_mat, axis=0)
        
        s1_v_rms = float(np.sqrt(np.mean(s1_v_mat_dyn**2)))
        s2_v_rms = float(np.sqrt(np.mean(s2_v_mat_dyn**2)))
        s1_s_rms = float(np.sqrt(np.mean(s1_s_mat_dyn**2)))
        s2_s_rms = float(np.sqrt(np.mean(s2_s_mat_dyn**2)))
        
        feats['s1_v_rms'] = s1_v_rms
        feats['s2_v_rms'] = s2_v_rms
        feats['v_rms_ratio'] = s1_v_rms / (s2_v_rms + 1e-8)
        feats['v_rms_diff'] = s1_v_rms - s2_v_rms
        
        feats['s1_s_rms'] = s1_s_rms
        feats['s2_s_rms'] = s2_s_rms
        feats['s_rms_ratio'] = s1_s_rms / (s2_s_rms + 1e-8)
        feats['s_rms_diff'] = s1_s_rms - s2_s_rms
        
        s1_v_ptp = float(np.mean(np.ptp(s1_v_mat_dyn, axis=0)))
        s2_v_ptp = float(np.mean(np.ptp(s2_v_mat_dyn, axis=0)))
        feats['v_ptp_ratio'] = s1_v_ptp / (s2_v_ptp + 1e-8)
        
        s1_s_kurt = float(np.mean(np.mean((s1_s_mat_dyn)**4, axis=0) / (np.var(s1_s_mat_dyn, axis=0)**2 + 1e-8)))
        s2_s_kurt = float(np.mean(np.mean((s2_s_mat_dyn)**4, axis=0) / (np.var(s2_s_mat_dyn, axis=0)**2 + 1e-8)))
        feats['s1_s_kurt'] = s1_s_kurt
        feats['s2_s_kurt'] = s2_s_kurt
        feats['s_kurt_diff'] = s1_s_kurt - s2_s_kurt
        
        car_v_diffs = []
        car_v_ratios = []
        for car in range(1, 9):
            c_s1 = [f"Vibration of bearing in position {p} of car {car}" for p in [1, 3, 5, 7]]
            c_s2 = [f"Vibration of bearing in position {p} of car {car}" for p in [2, 4, 6, 8]]
            cv1 = np.sqrt(np.mean((df[c_s1].values - np.mean(df[c_s1].values, axis=0))**2))
            cv2 = np.sqrt(np.mean((df[c_s2].values - np.mean(df[c_s2].values, axis=0))**2))
            diff = float(cv1 - cv2)
            ratio = float(cv1 / (cv2 + 1e-8))
            car_v_diffs.append(diff)
            car_v_ratios.append(ratio)
            feats[f'car{car}_v_diff'] = diff
            feats[f'car{car}_v_ratio'] = ratio
            
        feats['max_car_v_diff'] = float(np.max(car_v_diffs))
        feats['min_car_v_diff'] = float(np.min(car_v_diffs))
        feats['max_car_v_ratio'] = float(np.max(car_v_ratios))
        feats['min_car_v_ratio'] = float(np.min(car_v_ratios))
        feats['car_v_diff_range'] = feats['max_car_v_diff'] - feats['min_car_v_diff']
        
        mean_s1_sig = np.mean(s1_v_mat_dyn, axis=1)
        mean_s2_sig = np.mean(s2_v_mat_dyn, axis=1)
        f1, pxx1 = signal.welch(mean_s1_sig, fs=10000, nperseg=1024)
        f2, pxx2 = signal.welch(mean_s2_sig, fs=10000, nperseg=1024)
        
        bands = [
            ("band_0_100", 0, 100),
            ("band_100_250", 100, 250),
            ("band_250_500", 250, 500),
            ("band_500_1000", 500, 1000),
            ("band_1000_2000", 1000, 2000),
            ("band_2000_4000", 2000, 4000),
        ]
        for b_name, low, high in bands:
            idx = (f1 >= low) & (f1 < high)
            p1 = float(np.sum(pxx1[idx]))
            p2 = float(np.sum(pxx2[idx]))
            feats[f'{b_name}_s1'] = p1
            feats[f'{b_name}_s2'] = p2
            feats[f'{b_name}_ratio'] = float(p1 / (p2 + 1e-8))
            feats[f'{b_name}_diff'] = float(p1 - p2)
            
        if speed_info['speed_ms'] > 5.0:
            peak_idx1 = np.argmax(pxx1[f1 >= 100]) + int(100 / (10000 / 1024))
            peak_f1 = f1[peak_idx1]
            lambda1 = speed_info['speed_ms'] / (peak_f1 + 1e-6)
            feats['dominant_wavelength_s1'] = float(lambda1)
        else:
            feats['dominant_wavelength_s1'] = 0.0
            
        return feats
        
    def predict(self, df_or_filepath) -> dict:
        """
        Accepts a filepath (str) or a pandas DataFrame.
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
        
        # 1. Deterministic Low-Speed Gating
        if speed_kmh < self.low_speed_th:
            return {
                'prediction': 'Normal',
                'confidence': 1.0,
                'speed_kmh': speed_kmh,
                'status': 'Normal (Low / Stationary Speed)',
                'probabilities': {'Normal': 1.0, 'Side I': 0.0, 'Side II': 0.0},
                'diagnostic_summary': f"Train speed ({speed_kmh} km/h) is below dynamic excitation threshold (<10 km/h). Neither rail exhibits dynamic fault.",
                'asymmetry_ratio': round(feats['v_rms_ratio'], 3),
                'max_car_ratio': round(feats['max_car_v_ratio'], 3)
            }
            
        # 2. Model Inference
        x_vec = np.array([[feats[col] for col in self.feature_cols]])
        probs = self.model.predict_proba(x_vec)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = self.inv_label_map[pred_idx]
        confidence = float(probs[pred_idx])
        
        prob_dict = {
            'Normal': round(float(probs[self.label_map['Normal']]), 4),
            'Side I': round(float(probs[self.label_map['Side I']]), 4),
            'Side II': round(float(probs[self.label_map['Side II']]), 4),
        }
        
        status_desc = {
            'Normal': 'Normal Healthy Track',
            'Side I': 'Abnormal Corrugation Detected on Side I Rail',
            'Side II': 'Abnormal Corrugation Detected on Side II Rail'
        }
        
        return {
            'prediction': pred_label,
            'confidence': round(confidence, 4),
            'speed_kmh': speed_kmh,
            'status': status_desc.get(pred_label, pred_label),
            'probabilities': prob_dict,
            'diagnostic_summary': f"Model classified segment as {pred_label} at {speed_kmh} km/h with {round(confidence*100, 1)}% confidence.",
            'asymmetry_ratio': round(feats['v_rms_ratio'], 3),
            'max_car_ratio': round(feats['max_car_v_ratio'], 3)
        }

if __name__ == "__main__":
    # Self-test on a test file
    test_sample = os.path.join(os.path.dirname(__file__), "02_Datasets", "Rail_Corrugation", "Test", "Test1.csv")
    if os.path.exists(test_sample):
        predictor = RailPredictor()
        res = predictor.predict(test_sample)
        print("Self-Test Prediction on Test1.csv:")
        print(res)
