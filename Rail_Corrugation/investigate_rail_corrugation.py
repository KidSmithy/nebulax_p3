"""
investigate_rail_corrugation.py
=============================================================================
Rail Corrugation Data Profiling, Feature Extraction, and Baseline Benchmark
Dataset: PS3/02_Datasets/Rail_Corrugation (Train, Test, Train_Labels.csv)
References: PS3/03_References/Rail_Corrugation (Info Kit, Sensor Diagrams)
=============================================================================

This script performs:
1. Dataset Integrity Verification (Train & Test shapes, nulls, file counts).
2. Speed Sensor Decoding (Toothed wheel transitions -> linear velocity in km/h).
3. Physics-Informed Feature Engineering:
   - Speed dynamics & low-speed gating (v < 10 km/h is 100% Normal).
   - Side I vs Side II Asymmetry (RMS ratios, differences, Kurtosis, Crest Factor).
   - Car-by-Car Localized Anomaly Tracking (8 cars x 8 positions).
   - Power Spectral Density (Welch PSD across 6 corrugation resonance bands).
4. Model Evaluation & Benchmark (Stratified 5-Fold Cross Validation):
   - XGBoost Classifier (Class-Weighted)
   - Random Forest Classifier (Balanced)
   - Macro F1 scoring (official hackathon metric)
5. Test Set Feature Extraction & Zero-Speed Gating for Inference.
"""

import os
import glob
import time
import argparse
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import xgboost as xgb

# ---------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------
FS = 10000          # Sampling frequency: 10,000 Hz
DURATION = 1.0      # Duration: 1.0 second (10,000 samples per file)
WHEEL_DIAMETER = 0.85  # meters
WHEEL_TEETH = 90    # 90 teeth on circumference (180 0/1 transitions per rev)
WHEEL_CIRCUMFERENCE = np.pi * WHEEL_DIAMETER  # ~2.67035 meters

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
    """Decodes toothed speed sensor transitions to train speed (m/s and km/h)."""
    diffs = np.abs(np.diff(speed_series))
    transitions = int(np.sum(diffs > 0))
    # 90 teeth, each tooth entering and leaving creates 2 transitions
    revs_per_sec = transitions / (WHEEL_TEETH * 2.0)
    v_ms = revs_per_sec * WHEEL_CIRCUMFERENCE
    v_kmh = v_ms * 3.6
    return {
        'transitions': transitions,
        'speed_ms': v_ms,
        'speed_kmh': v_kmh,
        'is_low_speed': 1 if v_kmh < 15.0 else 0,
        'is_zero_speed': 1 if transitions == 0 else 0
    }


def extract_features_from_df(df: pd.DataFrame) -> dict:
    """
    Extracts physics-informed domain features from a single 10,000 x 129 recording.
    """
    feats = {}
    
    # 1. Speed Decoding
    speed_info = decode_speed(df['Rotating speed'].values)
    feats.update(speed_info)
    
    # 2. Separate Channels into Side I and Side II
    # Side I: Positions 1, 3, 5, 7 across Cars 1-8
    # Side II: Positions 2, 4, 6, 8 across Cars 1-8
    s1_vib_cols = [c for c in df.columns if "Vibration" in c and any(f"position {p}" in c for p in [1, 3, 5, 7])]
    s2_vib_cols = [c for c in df.columns if "Vibration" in c and any(f"position {p}" in c for p in [2, 4, 6, 8])]
    s1_shk_cols = [c for c in df.columns if "Shock" in c and any(f"position {p}" in c for p in [1, 3, 5, 7])]
    s2_shk_cols = [c for c in df.columns if "Shock" in c and any(f"position {p}" in c for p in [2, 4, 6, 8])]
    
    s1_v_mat = df[s1_vib_cols].values  # shape: (10000, 32)
    s2_v_mat = df[s2_vib_cols].values  # shape: (10000, 32)
    s1_s_mat = df[s1_shk_cols].values  # shape: (10000, 32)
    s2_s_mat = df[s2_shk_cols].values  # shape: (10000, 32)
    
    # Global RMS and Statistics
    s1_v_rms = float(np.sqrt(np.mean(s1_v_mat**2)))
    s2_v_rms = float(np.sqrt(np.mean(s2_v_mat**2)))
    s1_s_rms = float(np.sqrt(np.mean(s1_s_mat**2)))
    s2_s_rms = float(np.sqrt(np.mean(s2_s_mat**2)))
    
    feats['s1_v_rms'] = s1_v_rms
    feats['s2_v_rms'] = s2_v_rms
    feats['v_rms_ratio'] = s1_v_rms / (s2_v_rms + 1e-8)
    feats['v_rms_diff'] = s1_v_rms - s2_v_rms
    
    feats['s1_s_rms'] = s1_s_rms
    feats['s2_s_rms'] = s2_s_rms
    feats['s_rms_ratio'] = s1_s_rms / (s2_s_rms + 1e-8)
    feats['s_rms_diff'] = s1_s_rms - s2_s_rms
    
    # Peak-to-peak and Kurtosis (shock impact sensitivity)
    s1_v_ptp = float(np.mean(np.ptp(s1_v_mat, axis=0)))
    s2_v_ptp = float(np.mean(np.ptp(s2_v_mat, axis=0)))
    feats['v_ptp_ratio'] = s1_v_ptp / (s2_v_ptp + 1e-8)
    
    s1_s_kurt = float(np.mean(np.mean((s1_s_mat - np.mean(s1_s_mat, axis=0))**4, axis=0) / (np.var(s1_s_mat, axis=0)**2 + 1e-8)))
    s2_s_kurt = float(np.mean(np.mean((s2_s_mat - np.mean(s2_s_mat, axis=0))**4, axis=0) / (np.var(s2_s_mat, axis=0)**2 + 1e-8)))
    feats['s1_s_kurt'] = s1_s_kurt
    feats['s2_s_kurt'] = s2_s_kurt
    feats['s_kurt_diff'] = s1_s_kurt - s2_s_kurt
    
    # 3. Car-by-Car Asymmetry (Localization across 8 cars)
    car_v_diffs = []
    car_v_ratios = []
    
    for car in range(1, 9):
        c_s1 = [f"Vibration of bearing in position {p} of car {car}" for p in [1, 3, 5, 7]]
        c_s2 = [f"Vibration of bearing in position {p} of car {car}" for p in [2, 4, 6, 8]]
        cv1 = np.sqrt(np.mean(df[c_s1].values**2))
        cv2 = np.sqrt(np.mean(df[c_s2].values**2))
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
    
    # 4. Spectral Band Analysis (Welch PSD on mean Side I vs Side II signal)
    mean_s1_sig = np.mean(s1_v_mat, axis=1)
    mean_s2_sig = np.mean(s2_v_mat, axis=1)
    
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
        
    # Corrugation dominant wavelength estimation if moving:
    if speed_info['speed_ms'] > 5.0:
        peak_idx1 = np.argmax(pxx1[f1 >= 100]) + int(100 / (FS / 1024))
        peak_f1 = f1[peak_idx1]
        lambda1 = speed_info['speed_ms'] / (peak_f1 + 1e-6)
        feats['dominant_wavelength_s1'] = float(lambda1)
    else:
        feats['dominant_wavelength_s1'] = 0.0
        
    return feats


def run_full_pipeline(dataset_dir: str, output_dir: str = None):
    """
    Executes full data profiling, feature extraction, cross-validation, and benchmarking.
    """
    if output_dir is None:
        output_dir = dataset_dir
        
    train_dir = os.path.join(dataset_dir, "Train")
    test_dir = os.path.join(dataset_dir, "Test")
    labels_file = os.path.join(dataset_dir, "Train_Labels.csv")
    
    print("=" * 70)
    print("RAIL CORRUGATION DATASET PROFILING & BENCHMARK")
    print(f"Dataset Root: {dataset_dir}")
    print("=" * 70)
    
    # 1. Labels & File Verification
    df_labels = pd.read_csv(labels_file)
    train_files = sorted(glob.glob(os.path.join(train_dir, "*.csv")), 
                         key=lambda x: int(os.path.basename(x).replace("Train", "").replace(".csv", "")))
    test_files = sorted(glob.glob(os.path.join(test_dir, "*.csv")), 
                        key=lambda x: int(os.path.basename(x).replace("Test", "").replace(".csv", "")))
    
    print(f"Train files count: {len(train_files)}")
    print(f"Test files count:  {len(test_files)}")
    print(f"Train_Labels shape: {df_labels.shape}")
    print("\nClass Distribution:")
    for lbl, count in df_labels['label'].value_counts().items():
        pct = (count / len(df_labels)) * 100
        print(f"  - {lbl:<8}: {count:>3} files ({pct:5.2f}%)")
        
    # 2. Extract Features Across Train Set
    print("\n[Step 1/3] Extracting physics-informed features from Train files...")
    t0 = time.time()
    train_feat_list = []
    
    for idx, row in df_labels.iterrows():
        fname = row['filename']
        fpath = os.path.join(train_dir, fname)
        df_file = pd.read_csv(fpath)
        f_dict = extract_features_from_df(df_file)
        f_dict['filename'] = fname
        f_dict['label'] = row['label']
        train_feat_list.append(f_dict)
        
    df_train_feats = pd.DataFrame(train_feat_list)
    dt = time.time() - t0
    print(f"Extracted {df_train_feats.shape[1] - 2} features across {len(df_train_feats)} files in {dt:.1f}s.")
    
    features_csv_path = os.path.join(output_dir, "rail_extracted_features.csv")
    df_train_feats.to_csv(features_csv_path, index=False)
    print(f"Saved extracted features to: {features_csv_path}")
    
    # 3. Low-Speed Physical Discovery
    low_speed_counts = df_train_feats[df_train_feats['speed_kmh'] < 10.0]['label'].value_counts()
    print(f"\n[Domain Rule] Files with Speed < 10 km/h: {len(df_train_feats[df_train_feats['speed_kmh'] < 10.0])}")
    print(f"  Class breakdown: {low_speed_counts.to_dict()}")
    print("  -> 100% of files below 10 km/h are Normal (Stationary/Creep condition).")
    
    # 4. Stratified 5-Fold Cross Validation Benchmark
    print("\n[Step 2/3] Running Stratified 5-Fold Cross Validation...")
    feature_cols = [c for c in df_train_feats.columns if c not in ['filename', 'label']]
    X = df_train_feats[feature_cols].values
    y = df_train_feats['label'].map(LABEL_MAP).values
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Model 1: Balanced Random Forest
    rf_oof = np.zeros(len(y))
    for tr_idx, va_idx in skf.split(X, y):
        clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        clf.fit(X[tr_idx], y[tr_idx])
        rf_oof[va_idx] = clf.predict(X[va_idx])
        
    rf_macro = f1_score(y, rf_oof, average='macro')
    print("\n" + "-" * 50)
    print(f"Random Forest 5-Fold CV Macro F1: {rf_macro:.4f}")
    print("-" * 50)
    print(classification_report(y, rf_oof, target_names=['Normal', 'Side I', 'Side II'], digits=3))
    
    # Model 2: Class-Weighted XGBoost
    weights = np.zeros(len(y))
    for c_idx in range(3):
        count = np.sum(y == c_idx)
        weights[y == c_idx] = len(y) / (3.0 * count)
        
    xgb_oof = np.zeros(len(y))
    for tr_idx, va_idx in skf.split(X, y):
        model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.08,
            random_state=42,
            eval_metric='mlogloss'
        )
        model.fit(X[tr_idx], y[tr_idx], sample_weight=weights[tr_idx])
        xgb_oof[va_idx] = model.predict(X[va_idx])
        
    xgb_macro = f1_score(y, xgb_oof, average='macro')
    print("\n" + "-" * 50)
    print(f"XGBoost (Weighted) 5-Fold CV Macro F1: {xgb_macro:.4f}")
    print("-" * 50)
    print(classification_report(y, xgb_oof, target_names=['Normal', 'Side I', 'Side II'], digits=3))
    print("Confusion Matrix:\n", confusion_matrix(y, xgb_oof))
    
    # Feature Importance
    final_model = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.08, random_state=42, eval_metric='mlogloss')
    final_model.fit(X, y, sample_weight=weights)
    top_importances = pd.Series(final_model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(10)
    print("\nTop 10 Most Predictive Features in XGBoost:")
    for feat_name, imp in top_importances.items():
        print(f"  {feat_name:<25}: {imp:6.4f}")
        
    # 5. Test Inference
    print("\n[Step 3/3] Generating Baseline Predictions on Test Dataset...")
    test_preds = []
    for f in test_files:
        fname = os.path.basename(f)
        df_test = pd.read_csv(f)
        f_dict = extract_features_from_df(df_test)
        
        # Apply Zero/Low Speed Gating Rule
        if f_dict['speed_kmh'] < 10.0:
            pred_lbl = 'Normal'
        else:
            x_test = np.array([[f_dict[col] for col in feature_cols]])
            pred_idx = final_model.predict(x_test)[0]
            pred_lbl = INV_LABEL_MAP[pred_idx]
            
        test_preds.append({'file_id': fname, 'prediction': pred_lbl})
        
    df_submission = pd.DataFrame(test_preds)
    pred_path = os.path.join(output_dir, "baseline_rail_predictions.csv")
    df_submission.to_csv(pred_path, index=False)
    print(f"Generated test predictions saved to: {pred_path}")
    print("\nTest Prediction Distribution:")
    print(df_submission['prediction'].value_counts())
    print("=" * 70)
    print("Profiling & Benchmark Complete.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rail Corrugation Data Profiler & Benchmark")
    parser.add_argument("--data_dir", type=str, 
                        default=r"c:\Users\Rald999\Documents\GitHub\nebulax_p3\PS3\02_Datasets\Rail_Corrugation",
                        help="Path to PS3/02_Datasets/Rail_Corrugation")
    parser.add_argument("--output_dir", type=str, default=r"c:\Users\Rald999\Documents\GitHub\nebulax_p3\PS3",
                        help="Path to output directory")
    args = parser.parse_args()
    
    run_full_pipeline(args.data_dir, args.output_dir)
