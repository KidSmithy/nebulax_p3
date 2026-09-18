"""
Training Pipeline for SHM Subsystem
===================================
1. Loads training data and labels
2. Extracts domain & statistical features (or loads pre-extracted features for fast iteration)
3. Trains the winning Ensemble Model (Bayesian Ridge, ElasticNet, SVR, ExtraTrees)
4. Evaluates 5-fold CV score
5. Saves trained models & scaler to disk (shm_model.pkl)
"""

import os
import glob
import time
import pickle
import numpy as np
import pandas as pd
from scipy.signal import welch

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Data path resolution
DATA_CANDIDATES = [
    os.path.join(SCRIPT_DIR, "..", "PS3", "02_Datasets", "SHM"),
    os.path.join(SCRIPT_DIR, "02_Datasets", "SHM"),
    r"c:\Users\Rald999\Documents\GitHub\nebulax_p3\PS3\02_Datasets\SHM"
]

SHM_DATA_DIR = None
for cand in DATA_CANDIDATES:
    if os.path.exists(os.path.join(cand, "Train_Labels.csv")):
        SHM_DATA_DIR = os.path.abspath(cand)
        break

if SHM_DATA_DIR is None:
    raise FileNotFoundError("Could not find Train_Labels.csv in candidate locations.")

TRAIN_DIR = os.path.join(SHM_DATA_DIR, "Train")
LABEL_FILE = os.path.join(SHM_DATA_DIR, "Train_Labels.csv")
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "shm_model.pkl")
CACHE_FILE = os.path.join(SCRIPT_DIR, "investigation_output", "train_features_with_labels.csv")

FEATURE_NAMES = [
    'p2p', 'abs_peak', 'min', 'valley_min', 'psd_total_energy', 'mad',
    'rf_damage_proxy_m3.5', 'psd_energy_0_10Hz', 'rf_damage_proxy_m4.0',
    'rf_damage_proxy_m3.3', 'iqr', 'rf_damage_proxy_m4.5'
]

# --- Fast ASTM E1049-85 Rainflow Cycle Counter ---
def get_turning_points(series):
    diff1 = np.diff(series)
    turning = np.where(diff1[:-1] * diff1[1:] <= 0)[0] + 1
    return series[np.concatenate(([0], turning, [len(series) - 1]))]

def rainflow_fast(extrema):
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

def extract_file_features(filepath):
    sig = pd.read_csv(filepath, header=None, dtype=np.float32).values.flatten()
    mean_val = float(np.mean(sig))
    min_val = float(np.min(sig))
    max_val = float(np.max(sig))
    p2p_val = max_val - min_val
    abs_peak = float(np.max(np.abs(sig)))
    mad_val = float(np.mean(np.abs(sig - mean_val)))
    
    pct_vals = np.percentile(sig, [25, 75])
    iqr_val = float(pct_vals[1] - pct_vals[0])
    
    extrema = get_turning_points(sig)
    rf_ranges, rf_counts = rainflow_fast(extrema)
    amps = rf_ranges / 2.0
    
    rf_proxy_33 = float(np.sum(rf_counts * (amps ** 3.3)))
    rf_proxy_35 = float(np.sum(rf_counts * (amps ** 3.5)))
    rf_proxy_40 = float(np.sum(rf_counts * (amps ** 4.0)))
    rf_proxy_45 = float(np.sum(rf_counts * (amps ** 4.5)))
    
    freqs, psd = welch(sig, fs=256.0, nperseg=2048)
    psd_total = float(np.sum(psd))
    psd_0_10 = float(np.sum(psd[np.where((freqs >= 0) & (freqs < 10))[0]]))
    
    return [
        p2p_val, abs_peak, min_val, min_val, psd_total, mad_val,
        rf_proxy_35, psd_0_10, rf_proxy_40, rf_proxy_33, iqr_val, rf_proxy_45
    ]

def main():
    print("=" * 80)
    print(" TRAINING SHM PREDICTION MODEL ")
    print("=" * 80)

    # 1. Load Data / Features
    if os.path.exists(CACHE_FILE):
        print(f"Loading pre-computed training features from: {CACHE_FILE}")
        df_train = pd.read_csv(CACHE_FILE)
        X = df_train[FEATURE_NAMES].values
        y = df_train['damage'].values
    else:
        print(f"Extracting features from raw training files in {TRAIN_DIR}...")
        df_labels = pd.read_csv(LABEL_FILE)
        y_dict = dict(zip(df_labels['filename'], df_labels['damage']))
        train_files = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.csv")))
        X_list, y_list = [], []
        t0 = time.time()
        for i, f in enumerate(train_files):
            fname = os.path.basename(f)
            if fname in y_dict:
                X_list.append(extract_file_features(f))
                y_list.append(y_dict[fname])
            if (i + 1) % 16 == 0:
                print(f"  Processed {i+1}/{len(train_files)} files in {time.time()-t0:.1f}s")
        X = np.array(X_list)
        y = np.array(y_list)

    log_y = np.log(y)
    print(f"Dataset ready: {X.shape[0]} samples, {X.shape[1]} features.")

    # 2. Cross-Validation Assessment
    print("\nEvaluating 5-Fold Cross-Validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_preds = np.zeros(len(y))

    for tr_idx, val_idx in kf.split(X):
        scaler_fold = StandardScaler()
        X_tr = scaler_fold.fit_transform(X[tr_idx])
        X_val = scaler_fold.transform(X[val_idx])

        m1 = BayesianRidge().fit(X_tr, log_y[tr_idx])
        m2 = ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=42).fit(X_tr, log_y[tr_idx])
        m3 = SVR(C=1.0, epsilon=0.1).fit(X_tr, log_y[tr_idx])
        m4 = ExtraTreesRegressor(n_estimators=100, max_depth=4, random_state=42).fit(X_tr, log_y[tr_idx])

        p1 = np.exp(m1.predict(X_val))
        p2 = np.exp(m2.predict(X_val))
        p3 = np.exp(m3.predict(X_val))
        p4 = np.exp(m4.predict(X_val))

        fold_preds[val_idx] = 0.35 * p1 + 0.25 * p2 + 0.20 * p3 + 0.20 * p4

    cv_mape = np.mean(np.abs(y - fold_preds) / y)
    cv_score = max(0.0, 1.0 - cv_mape)
    print(f"  5-Fold CV Mean MAPE: {cv_mape:.4f} ({cv_mape*100:.2f}%)")
    print(f"  5-Fold CV Competition Score: {cv_score:.4f}")

    # 3. Train Final Model on 100% of Training Data
    print("\nFitting final ensemble on all training data...")
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)

    m1_final = BayesianRidge().fit(X_scaled, log_y)
    m2_final = ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=42).fit(X_scaled, log_y)
    m3_final = SVR(C=1.0, epsilon=0.1).fit(X_scaled, log_y)
    m4_final = ExtraTreesRegressor(n_estimators=100, max_depth=4, random_state=42).fit(X_scaled, log_y)

    model_payload = {
        'scaler': final_scaler,
        'features': FEATURE_NAMES,
        'weights': [0.35, 0.25, 0.20, 0.20],
        'models': [m1_final, m2_final, m3_final, m4_final],
        'clip_range': [0.025, 0.950],
        'cv_score': cv_score,
        'cv_mape': cv_mape
    }

    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(model_payload, f)

    print(f"\n[SUCCESS] Trained model artifact saved to: {MODEL_SAVE_PATH}")
    print(f"File size: {os.path.getsize(MODEL_SAVE_PATH) / 1024:.1f} KB")

if __name__ == "__main__":
    main()
