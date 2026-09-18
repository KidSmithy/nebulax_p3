"""
train_and_export_baseline.py
=============================================================================
Trains the Rail Corrugation Baseline Model and exports:
1. `models/rail_corrugation_xgb.json` (portable XGBoost model file)
2. `models/rail_model_bundle.joblib` (Model bundle containing feature columns,
   low-speed threshold, label mappings, and preprocessing metadata)
3. `models/rail_pipeline.py` (Ready-to-use Python inference API for frontend/backend)
=============================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import signal

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "02_Datasets", "Rail_Corrugation")
TRAIN_DIR = os.path.join(DATA_DIR, "Train")
LABELS_FILE = os.path.join(DATA_DIR, "Train_Labels.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Domain constants
FS = 10000
WHEEL_DIAMETER = 0.85
WHEEL_TEETH = 90
WHEEL_CIRC = np.pi * WHEEL_DIAMETER
LABEL_MAP = {'Normal': 0, 'Side I': 1, 'Side II': 2}
INV_LABEL_MAP = {0: 'Normal', 1: 'Side I', 2: 'Side II'}

SPECTRAL_BANDS = [
    ("band_0_100", 0, 100),
    ("band_100_250", 100, 250),
    ("band_250_500", 250, 500),
    ("band_500_1000", 500, 1000),
    ("band_1000_2000", 1000, 2000),
    ("band_2000_4000", 2000, 4000),
]

def decode_speed(speed_series: np.ndarray) -> dict:
    diffs = np.abs(np.diff(speed_series))
    transitions = int(np.sum(diffs > 0))
    revs = transitions / (WHEEL_TEETH * 2.0)
    v_ms = revs * WHEEL_CIRC
    v_kmh = v_ms * 3.6
    return {
        'transitions': transitions,
        'speed_ms': v_ms,
        'speed_kmh': v_kmh,
        'is_low_speed': 1 if v_kmh < 15.0 else 0,
        'is_zero_speed': 1 if transitions == 0 else 0
    }

def extract_features_from_df(df: pd.DataFrame) -> dict:
    feats = {}
    speed_info = decode_speed(df['Rotating speed'].values)
    feats.update(speed_info)
    
    s1_vib_cols = [c for c in df.columns if "Vibration" in c and any(f"position {p}" in c for p in [1, 3, 5, 7])]
    s2_vib_cols = [c for c in df.columns if "Vibration" in c and any(f"position {p}" in c for p in [2, 4, 6, 8])]
    s1_shk_cols = [c for c in df.columns if "Shock" in c and any(f"position {p}" in c for p in [1, 3, 5, 7])]
    s2_shk_cols = [c for c in df.columns if "Shock" in c and any(f"position {p}" in c for p in [2, 4, 6, 8])]
    
    # DC offset removal (mean subtraction for dynamic acceleration)
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
    f1, pxx1 = signal.welch(mean_s1_sig, fs=FS, nperseg=1024)
    f2, pxx2 = signal.welch(mean_s2_sig, fs=FS, nperseg=1024)
    
    for b_name, low, high in SPECTRAL_BANDS:
        idx = (f1 >= low) & (f1 < high)
        p1 = float(np.sum(pxx1[idx]))
        p2 = float(np.sum(pxx2[idx]))
        feats[f'{b_name}_s1'] = p1
        feats[f'{b_name}_s2'] = p2
        feats[f'{b_name}_ratio'] = float(p1 / (p2 + 1e-8))
        feats[f'{b_name}_diff'] = float(p1 - p2)
        
    if speed_info['speed_ms'] > 5.0:
        peak_idx1 = np.argmax(pxx1[f1 >= 100]) + int(100 / (FS / 1024))
        peak_f1 = f1[peak_idx1]
        lambda1 = speed_info['speed_ms'] / (peak_f1 + 1e-6)
        feats['dominant_wavelength_s1'] = float(lambda1)
    else:
        feats['dominant_wavelength_s1'] = 0.0
        
    return feats

def main():
    print("Loading training dataset and labels...")
    df_labels = pd.read_csv(LABELS_FILE)
    
    # Check if pre-extracted features exist, else extract
    features_csv = os.path.join(BASE_DIR, "rail_extracted_features.csv")
    if os.path.exists(features_csv):
        print(f"Loading pre-extracted features from {features_csv}...")
        df_feats = pd.read_csv(features_csv)
    else:
        print("Extracting features from 272 raw CSV files...")
        feat_list = []
        for idx, row in df_labels.iterrows():
            fname = row['filename']
            fpath = os.path.join(TRAIN_DIR, fname)
            f_dict = extract_features_from_df(pd.read_csv(fpath))
            f_dict['filename'] = fname
            f_dict['label'] = row['label']
            feat_list.append(f_dict)
        df_feats = pd.DataFrame(feat_list)
        df_feats.to_csv(features_csv, index=False)
        
    feature_cols = [c for c in df_feats.columns if c not in ['filename', 'label']]
    X = df_feats[feature_cols].values
    y = df_feats['label'].map(LABEL_MAP).values
    
    # Class weights for handling 86% Normal imbalance
    weights = np.zeros(len(y))
    for c_idx in range(3):
        count = np.sum(y == c_idx)
        weights[y == c_idx] = len(y) / (3.0 * count)
        
    print(f"Training production XGBoost baseline on all {len(y)} samples ({len(feature_cols)} features)...")
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.08,
        random_state=42,
        eval_metric='mlogloss'
    )
    model.fit(X, y, sample_weight=weights)
    
    # Export 1: XGBoost native JSON format
    json_path = os.path.join(MODELS_DIR, "rail_corrugation_xgb.json")
    model.save_model(json_path)
    print(f"[Exported] Native XGBoost model saved to: {json_path}")
    
    # Export 2: Self-contained Joblib Bundle
    bundle = {
        'model': model,
        'feature_cols': feature_cols,
        'label_map': LABEL_MAP,
        'inv_label_map': INV_LABEL_MAP,
        'speed_gating_threshold_kmh': 10.0,
        'metadata': {
            'model_type': 'XGBClassifier',
            'version': '1.0.0-baseline',
            'training_samples': len(y),
            'features_count': len(feature_cols),
            'target_metric': 'Macro F1'
        }
    }
    bundle_path = os.path.join(MODELS_DIR, "rail_model_bundle.joblib")
    joblib.dump(bundle, bundle_path)
    print(f"[Exported] Full model bundle saved to: {bundle_path}")
    
    print("\nModel successfully exported! Ready for frontend consumption.")

if __name__ == "__main__":
    main()
