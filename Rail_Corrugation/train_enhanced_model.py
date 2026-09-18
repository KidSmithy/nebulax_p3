"""
train_enhanced_model.py
=============================================================================
Enhanced Model Development for Rail Corrugation Monitoring
Implements the top recommendations from expert peer review:

1. Enhanced Feature Engineering:
   - Robust order statistics across 8 cars: Top-1, Top-2, Top-3 car asymmetry,
     mean of top-2, mean of top-3, and car variance.
   - Speed-normalized spatial wavelength bands (lambda = v / f in [2-4cm, 4-7cm, 7-12cm, 12-20cm]).
   - Spectral centroid & spectral entropy differences between Side I and Side II.
   - Channel-wise DC offset removal (mean subtraction).
2. Clean Physical Low-Speed Gate:
   - Fixed at 10.0 km/h (physically grounded threshold).
3. Repeated Stratified Cross-Validation:
   - 5 Folds x 5 Seeds (25 evaluation splits) for stable evaluation of rare Side I class.
4. Out-of-Fold (OOF) Decision Threshold Optimization:
   - Calibrated probability divisors specifically maximizing Macro F1.
5. Production Model Bundle & Submission Export.
=============================================================================
"""

import os
import time
import joblib
import warnings
import numpy as np
import pandas as pd
from scipy import signal
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(r"c:\Users\Rald999\Documents\GitHub\nebulax_p3\PS3\02_Datasets\Rail_Corrugation")
TRAIN_DIR = os.path.join(DATA_DIR, "Train")
TEST_DIR = os.path.join(DATA_DIR, "Test")
LABELS_FILE = os.path.join(DATA_DIR, "Train_Labels.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

FS = 10000
WHEEL_DIAMETER = 0.85
WHEEL_CIRC = np.pi * WHEEL_DIAMETER
LABEL_MAP = {'Normal': 0, 'Side I': 1, 'Side II': 2}
INV_MAP = {0: 'Normal', 1: 'Side I', 2: 'Side II'}

def decode_speed(speed_series: np.ndarray) -> dict:
    diffs = np.abs(np.diff(speed_series))
    transitions = int(np.sum(diffs > 0))
    revs = transitions / 180.0
    v_ms = revs * WHEEL_CIRC
    v_kmh = v_ms * 3.6
    return {
        'transitions': transitions,
        'speed_ms': v_ms,
        'speed_kmh': v_kmh,
        'is_low_speed': 1 if v_kmh < 10.0 else 0
    }

def spectral_centroid_and_entropy(pxx, freqs):
    # Centroid
    sum_pxx = np.sum(pxx) + 1e-12
    centroid = np.sum(freqs * pxx) / sum_pxx
    # Entropy
    p_norm = pxx / sum_pxx
    p_norm = p_norm[p_norm > 0]
    entropy = -np.sum(p_norm * np.log2(p_norm))
    return float(centroid), float(entropy)

def extract_advanced_features(df: pd.DataFrame) -> dict:
    feats = {}
    speed_info = decode_speed(df['Rotating speed'].values)
    feats.update(speed_info)
    v_ms = speed_info['speed_ms']
    
    # 1. Identify Channels
    s1_vib_cols = [c for c in df.columns if "Vibration" in c and any(f"position {p}" in c for p in [1, 3, 5, 7])]
    s2_vib_cols = [c for c in df.columns if "Vibration" in c and any(f"position {p}" in c for p in [2, 4, 6, 8])]
    s1_shk_cols = [c for c in df.columns if "Shock" in c and any(f"position {p}" in c for p in [1, 3, 5, 7])]
    s2_shk_cols = [c for c in df.columns if "Shock" in c and any(f"position {p}" in c for p in [2, 4, 6, 8])]
    
    # 2. DC Offset Removal (Mean Subtraction)
    s1_v_mat = df[s1_vib_cols].values - np.mean(df[s1_vib_cols].values, axis=0)
    s2_v_mat = df[s2_vib_cols].values - np.mean(df[s2_vib_cols].values, axis=0)
    s1_s_mat = df[s1_shk_cols].values - np.mean(df[s1_shk_cols].values, axis=0)
    s2_s_mat = df[s2_shk_cols].values - np.mean(df[s2_shk_cols].values, axis=0)
    
    # 3. Global Energy & Ratios
    s1_v_rms = float(np.sqrt(np.mean(s1_v_mat**2)))
    s2_v_rms = float(np.sqrt(np.mean(s2_v_mat**2)))
    feats['v_rms_ratio'] = s1_v_rms / (s2_v_rms + 1e-8)
    feats['v_rms_diff'] = s1_v_rms - s2_v_rms
    
    s1_s_rms = float(np.sqrt(np.mean(s1_s_mat**2)))
    s2_s_rms = float(np.sqrt(np.mean(s2_s_mat**2)))
    feats['s_rms_ratio'] = s1_s_rms / (s2_s_rms + 1e-8)
    feats['s_rms_diff'] = s1_s_rms - s2_s_rms
    
    # Shock Kurtosis & Crest Factor
    s1_kurt = float(np.mean(np.mean(s1_s_mat**4, axis=0) / (np.var(s1_s_mat, axis=0)**2 + 1e-8)))
    s2_kurt = float(np.mean(np.mean(s2_s_mat**4, axis=0) / (np.var(s2_s_mat, axis=0)**2 + 1e-8)))
    feats['s_kurt_ratio'] = s1_kurt / (s2_kurt + 1e-8)
    feats['s_kurt_diff'] = s1_kurt - s2_kurt
    
    # 4. Car-by-Car Localized Order Statistics (Feedback Priority 3)
    car_v_ratios = []
    car_v_diffs = []
    
    for car in range(1, 9):
        c_s1 = [f"Vibration of bearing in position {p} of car {car}" for p in [1, 3, 5, 7]]
        c_s2 = [f"Vibration of bearing in position {p} of car {car}" for p in [2, 4, 6, 8]]
        
        cv1 = np.sqrt(np.mean((df[c_s1].values - np.mean(df[c_s1].values, axis=0))**2))
        cv2 = np.sqrt(np.mean((df[c_s2].values - np.mean(df[c_s2].values, axis=0))**2))
        
        r = float(cv1 / (cv2 + 1e-8))
        d = float(cv1 - cv2)
        car_v_ratios.append(r)
        car_v_diffs.append(d)
        feats[f'car{car}_v_ratio'] = r
        feats[f'car{car}_v_diff'] = d
        
    sorted_ratios = np.sort(car_v_ratios) # Ascending
    sorted_diffs = np.sort(car_v_diffs)
    
    # Side I asymmetry peaks (large ratios/diffs)
    feats['top1_car_ratio'] = float(sorted_ratios[-1])
    feats['top2_car_ratio'] = float(sorted_ratios[-2])
    feats['top3_car_ratio'] = float(sorted_ratios[-3])
    feats['mean_top2_car_ratio'] = float((sorted_ratios[-1] + sorted_ratios[-2]) / 2.0)
    feats['mean_top3_car_ratio'] = float((sorted_ratios[-1] + sorted_ratios[-2] + sorted_ratios[-3]) / 3.0)
    
    feats['top1_car_diff'] = float(sorted_diffs[-1])
    feats['top2_car_diff'] = float(sorted_diffs[-2])
    feats['mean_top2_car_diff'] = float((sorted_diffs[-1] + sorted_diffs[-2]) / 2.0)
    
    # Side II asymmetry peaks (small/negative ratios and diffs)
    feats['min1_car_ratio'] = float(sorted_ratios[0])
    feats['min2_car_ratio'] = float(sorted_ratios[1])
    feats['mean_min2_car_ratio'] = float((sorted_ratios[0] + sorted_ratios[1]) / 2.0)
    feats['min1_car_diff'] = float(sorted_diffs[0])
    feats['min2_car_diff'] = float(sorted_diffs[1])
    feats['mean_min2_car_diff'] = float((sorted_diffs[0] + sorted_diffs[1]) / 2.0)
    
    feats['car_ratio_std'] = float(np.std(car_v_ratios))
    feats['car_ratio_range'] = float(sorted_ratios[-1] - sorted_ratios[0])
    
    # 5. Welch Spectral Power and Centroid/Entropy
    mean_s1_sig = np.mean(s1_v_mat, axis=1)
    mean_s2_sig = np.mean(s2_v_mat, axis=1)
    f, pxx1 = signal.welch(mean_s1_sig, fs=FS, nperseg=1024)
    f, pxx2 = signal.welch(mean_s2_sig, fs=FS, nperseg=1024)
    
    cent1, ent1 = spectral_centroid_and_entropy(pxx1, f)
    cent2, ent2 = spectral_centroid_and_entropy(pxx2, f)
    feats['spectral_centroid_diff'] = cent1 - cent2
    feats['spectral_entropy_diff'] = ent1 - ent2
    
    # Fixed frequency bands
    fixed_bands = [
        ("band_100_250", 100, 250),
        ("band_250_500", 250, 500),
        ("band_500_1000", 500, 1000),
        ("band_1000_2000", 1000, 2000),
        ("band_2000_4000", 2000, 4000)
    ]
    for bname, low, high in fixed_bands:
        idx = (f >= low) & (f < high)
        p1 = float(np.sum(pxx1[idx]))
        p2 = float(np.sum(pxx2[idx]))
        feats[f'{bname}_ratio'] = float(p1 / (p2 + 1e-8))
        feats[f'{bname}_diff'] = float(p1 - p2)
        feats[f'{bname}_s1'] = p1
        feats[f'{bname}_s2'] = p2
        
    # 6. Speed-Normalized Spatial Wavelength Bins (Feedback Priority 4)
    # f = v / lambda  ==>  lambda = v / f
    if v_ms > 3.0:
        wavelength_bins = [
            ("wave_2_4cm", 0.02, 0.04),
            ("wave_4_7cm", 0.04, 0.07),
            ("wave_7_12cm", 0.07, 0.12),
            ("wave_12_20cm", 0.12, 0.20),
            ("wave_20_30cm", 0.20, 0.30)
        ]
        for wname, lam_min, lam_max in wavelength_bins:
            # Frequency is inversely proportional: f_min = v / lam_max, f_max = v / lam_min
            f_low = v_ms / lam_max
            f_high = v_ms / lam_min
            idx_w = (f >= f_low) & (f < f_high)
            pw1 = float(np.sum(pxx1[idx_w]))
            pw2 = float(np.sum(pxx2[idx_w]))
            feats[f'{wname}_ratio'] = float(pw1 / (pw2 + 1e-8))
            feats[f'{wname}_diff'] = float(pw1 - pw2)
    else:
        for wname in ["wave_2_4cm", "wave_4_7cm", "wave_7_12cm", "wave_12_20cm", "wave_20_30cm"]:
            feats[f'{wname}_ratio'] = 1.0
            feats[f'{wname}_diff'] = 0.0
            
    return feats

def main():
    print("=" * 80)
    print("ENHANCED RAIL CORRUGATION MODEL DEVELOPMENT")
    print("=" * 80)
    
    # 1. Feature Extraction / Caching
    enhanced_csv = os.path.join(BASE_DIR, "rail_enhanced_features.csv")
    df_labels = pd.read_csv(LABELS_FILE)
    
    if os.path.exists(enhanced_csv):
        print(f"Loading existing enhanced features from {enhanced_csv}...")
        df_feats = pd.read_csv(enhanced_csv)
    else:
        print(f"Extracting advanced features from {len(df_labels)} Train files...")
        t0 = time.time()
        records = []
        for idx, row in df_labels.iterrows():
            fname = row['filename']
            fpath = os.path.join(TRAIN_DIR, fname)
            df_file = pd.read_csv(fpath)
            f_dict = extract_advanced_features(df_file)
            f_dict['filename'] = fname
            f_dict['label'] = row['label']
            records.append(f_dict)
        df_feats = pd.DataFrame(records)
        df_feats.to_csv(enhanced_csv, index=False)
        print(f"Features extracted in {time.time()-t0:.1f}s. Saved to {enhanced_csv}")
        
    feature_cols = [c for c in df_feats.columns if c not in ['filename', 'label']]
    print(f"Total Enhanced Features: {len(feature_cols)}")
    
    X = df_feats[feature_cols].values
    y = df_feats['label'].map(LABEL_MAP).values
    speed = df_feats['speed_kmh'].values
    
    # Class weights for minority handling
    w_side_i = 18.0
    w_side_ii = 7.5
    class_weights = np.array([1.0, w_side_i, w_side_ii])
    sample_weights = class_weights[y]

    # -------------------------------------------------------------------------
    # 2. Repeated Stratified Cross Validation (5 Folds x 5 Seeds = 25 evaluations)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RUNNING REPEATED STRATIFIED CROSS-VALIDATION (5 Folds x 5 Random Seeds)")
    print("=" * 80)
    
    SEEDS = [42, 101, 2024, 777, 999]
    models_to_test = ["LightGBMLarge", "CatBoost", "XGBoost", "EnsembleBlend"]
    results_records = []
    
    oof_predictions_all = {m: np.zeros((len(SEEDS), len(y), 3)) for m in models_to_test}
    
    for s_idx, seed in enumerate(SEEDS):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        
        for tr_idx, va_idx in skf.split(X, y):
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_va, y_va = X[va_idx], y[va_idx]
            sw_tr = sample_weights[tr_idx]
            
            # 1. LightGBM
            lgb_clf = lgb.LGBMClassifier(
                n_estimators=160,
                num_leaves=15,
                max_depth=5,
                learning_rate=0.06,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=seed,
                verbose=-1
            )
            lgb_clf.fit(X_tr, y_tr, sample_weight=sw_tr)
            p_lgb = lgb_clf.predict_proba(X_va)
            oof_predictions_all["LightGBMLarge"][s_idx, va_idx] = p_lgb
            
            # 2. CatBoost
            cb_clf = CatBoostClassifier(
                iterations=160,
                depth=6,
                learning_rate=0.08,
                l2_leaf_reg=3.0,
                loss_function="MultiClass",
                class_weights=class_weights.tolist(),
                random_seed=seed,
                verbose=0
            )
            cb_clf.fit(X_tr, y_tr)
            p_cb = cb_clf.predict_proba(X_va)
            oof_predictions_all["CatBoost"][s_idx, va_idx] = p_cb
            
            # 3. XGBoost
            xgb_clf = xgb.XGBClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.07,
                colsample_bytree=0.85,
                subsample=0.85,
                reg_lambda=2.0,
                random_state=seed,
                eval_metric="mlogloss",
                verbosity=0
            )
            xgb_clf.fit(X_tr, y_tr, sample_weight=sw_tr)
            p_xgb = xgb_clf.predict_proba(X_va)
            oof_predictions_all["XGBoost"][s_idx, va_idx] = p_xgb
            
            # 4. Ensemble Blend (0.45 LightGBM + 0.35 CatBoost + 0.20 XGBoost)
            p_ens = 0.45 * p_lgb + 0.35 * p_cb + 0.20 * p_xgb
            oof_predictions_all["EnsembleBlend"][s_idx, va_idx] = p_ens
            
    # Compute stability metrics across 5 seeds
    print("\n--- Repeated CV Performance across 5 Seeds (Default Argmax + 10km/h Gate) ---")
    for m in models_to_test:
        f1_list = []
        s1_f1_list = []
        s2_f1_list = []
        for s_idx in range(len(SEEDS)):
            probs = oof_predictions_all[m][s_idx]
            preds = np.argmax(probs, axis=1)
            preds[speed < 10.0] = 0  # 10 km/h physics gate
            f1_list.append(f1_score(y, preds, average='macro'))
            per_class = f1_score(y, preds, average=None)
            s1_f1_list.append(per_class[1])
            s2_f1_list.append(per_class[2])
            
        print(f"Model: {m:<15} | Macro F1: {np.mean(f1_list):.4f} +/- {np.std(f1_list):.4f} (Min: {np.min(f1_list):.4f}) | Side I F1: {np.mean(s1_f1_list):.4f} | Side II F1: {np.mean(s2_f1_list):.4f}")

    # -------------------------------------------------------------------------
    # 3. Decision Threshold Optimization for Macro F1 (Feedback Priority 1)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 3: Decision Threshold Optimization on Repeated OOF Probabilities")
    print("=" * 80)
    
    # Average OOF probability across all 5 seeds for Ensemble
    mean_oof_probs = np.mean(oof_predictions_all["EnsembleBlend"], axis=0)
    
    # Grid search for optimal probability divisors
    best_tuned_f1 = 0.0
    best_divisors = (1.0, 1.0, 1.0)
    
    for div_s1 in np.linspace(0.18, 0.45, 28):
        for div_s2 in np.linspace(0.70, 1.10, 17):
            divs = np.array([1.0, div_s1, div_s2])
            adj_p = mean_oof_probs / divs
            preds = np.argmax(adj_p, axis=1)
            preds[speed < 10.0] = 0
            score = f1_score(y, preds, average='macro')
            if score > best_tuned_f1:
                best_tuned_f1 = score
                best_divisors = divs
                
    adj_p_best = mean_oof_probs / best_divisors
    final_oof_preds = np.argmax(adj_p_best, axis=1)
    final_oof_preds[speed < 10.0] = 0
    
    print(f"Optimal Threshold Divisors [Normal, Side I, Side II]: {np.round(best_divisors, 3)}")
    print(f"FINAL TUNED ENSEMBLE MACRO F1: {best_tuned_f1:.4f}")
    print("\nFinal Out-Of-Fold Classification Report:")
    print(classification_report(y, final_oof_preds, target_names=['Normal', 'Side I', 'Side II'], digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y, final_oof_preds))

    # -------------------------------------------------------------------------
    # 4. Train Final Production Models on 100% of Training Data
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TRAINING FINAL ENSEMBLE PRODUCTION MODELS (Full 272 Recordings)")
    print("=" * 80)
    
    final_lgb = lgb.LGBMClassifier(n_estimators=160, num_leaves=15, max_depth=5, learning_rate=0.06, subsample=0.85, colsample_bytree=0.85, random_state=42, verbose=-1)
    final_lgb.fit(X, y, sample_weight=sample_weights)
    
    final_cb = CatBoostClassifier(iterations=160, depth=6, learning_rate=0.08, l2_leaf_reg=3.0, loss_function="MultiClass", class_weights=class_weights.tolist(), random_seed=42, verbose=0)
    final_cb.fit(X, y)
    
    final_xgb = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.07, colsample_bytree=0.85, subsample=0.85, reg_lambda=2.0, random_state=42, eval_metric="mlogloss", verbosity=0)
    final_xgb.fit(X, y, sample_weight=sample_weights)
    
    production_bundle = {
        'models': {
            'lgb': final_lgb,
            'catboost': final_cb,
            'xgboost': final_xgb
        },
        'blend_weights': (0.45, 0.35, 0.20),
        'feature_cols': feature_cols,
        'threshold_divisors': best_divisors,
        'low_speed_gate_kmh': 10.0,
        'label_map': LABEL_MAP,
        'inv_label_map': INV_MAP,
        'performance_metrics': {
            'oof_macro_f1': best_tuned_f1,
            'side_i_f1': float(f1_score(y, final_oof_preds, average=None)[1]),
            'side_ii_f1': float(f1_score(y, final_oof_preds, average=None)[2]),
            'normal_f1': float(f1_score(y, final_oof_preds, average=None)[0]),
            'accuracy': float(np.mean(y == final_oof_preds))
        }
    }
    
    bundle_dest = os.path.join(MODELS_DIR, "rail_model_bundle.joblib")
    joblib.dump(production_bundle, bundle_dest)
    print(f"Updated production model bundle saved to: {bundle_dest}")

    # -------------------------------------------------------------------------
    # 5. Generate Test Predictions
    # -------------------------------------------------------------------------
    test_files = sorted(
        [f for f in os.listdir(TEST_DIR) if f.endswith(".csv")],
        key=lambda x: int(x.replace("Test", "").replace(".csv", ""))
    )
    
    print(f"\nGenerating predictions on {len(test_files)} test files...")
    test_results = []
    
    for tf in test_files:
        df_t = pd.read_csv(os.path.join(TEST_DIR, tf))
        feats_t = extract_advanced_features(df_t)
        
        if feats_t['speed_kmh'] < 10.0:
            pred_lbl = "Normal"
        else:
            x_t = np.array([[feats_t[c] for c in feature_cols]])
            p1 = final_lgb.predict_proba(x_t)[0]
            p2 = final_cb.predict_proba(x_t)[0]
            p3 = final_xgb.predict_proba(x_t)[0]
            p_ens = 0.45 * p1 + 0.35 * p2 + 0.20 * p3
            
            adj_p = p_ens / best_divisors
            pred_lbl = INV_MAP[np.argmax(adj_p)]
            
        test_results.append({'file_id': tf, 'prediction': pred_lbl})
        
    df_sub = pd.DataFrame(test_results)
    pred_path = os.path.join(BASE_DIR, "automl_rail_predictions.csv")
    df_sub.to_csv(pred_path, index=False)
    df_sub.to_csv(os.path.join(BASE_DIR, "baseline_rail_predictions.csv"), index=False)
    print(f"Test predictions updated in {pred_path}")
    print("\nTest Prediction Distribution:")
    print(df_sub['prediction'].value_counts())
    print("\n" + "=" * 80)
    print("ENHANCED MODEL DEVELOPMENT COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
