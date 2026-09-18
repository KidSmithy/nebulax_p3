"""
Train & Export ACV Production Model Bundle
-----------------------------------------
Saves a complete, self-contained model bundle (.joblib) for frontend UI / web app integration.
Includes:
- Trained Regularized Pairwise Logistic Regression model
- Physics Health-Index weights and scaling parameters
- Feature definitions and column normalizers
- Cleaning & outlier thresholds
- Model metadata and versioning
"""

import os
import sys
import glob
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

sys.stdout.reconfigure(encoding='utf-8')

ROOT_DIR = r"c:\Users\Rald999\Documents\GitHub\nebulax_p3"
PS3_DIR = os.path.join(ROOT_DIR, "PS3")
ACV_DIR = os.path.join(ROOT_DIR, "ACV")
MODELS_DIR = os.path.join(PS3_DIR, "models")
DATA_DIR = os.path.join(PS3_DIR, "02_Datasets", "ACV")
TRAIN_DIR = os.path.join(DATA_DIR, "Train")

def clean_and_preprocess_case(df):
    rename_dict = {}
    for col in df.columns:
        if "Outside Temperature Sensor Reading" in col:
            rename_dict[col] = col.replace("Outside Temperature Sensor Reading", "Outdoor Average Temperature")
        elif "Fresh Air Temperature Detected Value" in col:
            rename_dict[col] = col.replace("Fresh Air Temperature Detected Value", "Outdoor Average Temperature")
        elif "Passenger Cabin Temperature Detected Value" in col:
            rename_dict[col] = col.replace("Passenger Cabin Temperature Detected Value", "Indoor Average Temperature")
        elif "Target Temperature Value" in col:
            rename_dict[col] = col.replace("Target Temperature Value", "ACV Control Temperature (Cooling)")
    df = df.rename(columns=rename_dict)
    
    car_cols = [c for c in df.columns if c.startswith("Car ") and c != "Car model"]
    all_cids = sorted(list(set(c.split(" - ")[0].replace("Car ", "").strip() for c in car_cols)))
    active_cars = [cid for cid in all_cids if f"Car {cid} - Indoor Average Temperature" in df.columns 
                   and df[f"Car {cid} - Indoor Average Temperature"].notnull().sum() > 0]
                   
    for cid in active_cars:
        t_col = f"Car {cid} - Indoor Average Temperature"
        valid_col = f"Car {cid} - ACV Information Valid"
        if valid_col in df.columns:
            invalid_mask = df[valid_col].astype(str).str.contains("Invalid", na=False)
            df.loc[invalid_mask, t_col] = np.nan
        if t_col in df.columns:
            s = pd.to_numeric(df[t_col], errors='coerce')
            s = s.mask((s <= 5.0) | (s >= 45.0), np.nan)
            df[t_col] = s.ffill().bfill()
            
    active_temp_cols = [f"Car {cid} - Indoor Average Temperature" for cid in active_cars if f"Car {cid} - Indoor Average Temperature" in df.columns]
    df = df.dropna(subset=active_temp_cols, how='all').reset_index(drop=True)
    return df, active_cars

def extract_features(df, active_cars):
    indoor_dict = {}
    cooling_set_dict = {}
    for c in active_cars:
        tin_col = f"Car {c} - Indoor Average Temperature"
        tset_col = f"Car {c} - ACV Control Temperature (Cooling)"
        if tin_col in df.columns:
            indoor_dict[c] = df[tin_col]
        if tset_col in df.columns:
            cooling_set_dict[c] = pd.to_numeric(df[tset_col], errors='coerce')
            
    indoor_df = pd.DataFrame(indoor_dict)
    cooling_set_df = pd.DataFrame(cooling_set_dict)
    median_indoor = indoor_df.median(axis=1)
    
    features = {}
    for c in active_cars:
        tin = indoor_df[c]
        rel_diff = tin - median_indoor
        tset = cooling_set_df[c] if c in cooling_set_df.columns else pd.Series(np.nan, index=df.index)
        t_minus_set = tin - tset
        
        mode_col = f"Car {c} - ACV Running Mode"
        if mode_col in df.columns:
            is_cooling = df[mode_col].astype(str).str.contains("Cooling", na=False)
        else:
            is_cooling = pd.Series(True, index=df.index)
            
        rel_cooling = rel_diff[is_cooling]
        t_minus_set_cooling = t_minus_set[is_cooling]
        
        rolling_elevated = (rel_diff.rolling(window=30, min_periods=10).mean() > 0.20)
        persistence_frac = float(rolling_elevated[is_cooling].mean()) if len(rolling_elevated[is_cooling].dropna()) > 0 else 0.0
        
        feat = {
            "mean_indoor": float(np.nanmean(tin)),
            "mean_rel_diff_cooling": float(np.nanmean(rel_cooling)) if len(rel_cooling.dropna()) > 0 else float(np.nanmean(rel_diff)),
            "mean_t_minus_set_cooling": float(np.nanmean(t_minus_set_cooling)) if len(t_minus_set_cooling.dropna()) > 0 else 0.0,
            "p95_rel_diff": float(np.nanpercentile(rel_diff.dropna(), 95)) if len(rel_diff.dropna()) > 0 else 0.0,
            "frac_above_median": float((rel_diff > 0.20).mean()),
            "persistence_15m": persistence_frac
        }
        features[c] = feat
        
    df_feat = pd.DataFrame(features).T
    df_feat_z = df_feat.copy()
    raw_cols = ["mean_rel_diff_cooling", "mean_t_minus_set_cooling", "p95_rel_diff", "frac_above_median", "persistence_15m"]
    for col in raw_cols:
        std_val = df_feat[col].std()
        df_feat_z[f"{col}_z"] = (df_feat[col] - df_feat[col].mean()) / (std_val if std_val > 1e-6 else 1.0)
        
    return df_feat, df_feat_z

def main():
    print("=" * 80)
    print("TRAINING & EXPORTING ACV MODEL BUNDLE FOR FRONTEND INTEGRATION")
    print("=" * 80)
    
    labels_file = os.path.join(DATA_DIR, "Train_Labels.csv")
    df_labels = pd.read_csv(labels_file)
    label_map = {row['filename']: str(row['faulty_car']).zfill(2) for _, row in df_labels.iterrows()}
    
    train_files = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.xlsx")))
    train_feat_z_dict = {}
    
    for fpath in train_files:
        fname = os.path.basename(fpath)
        df = pd.read_excel(fpath)
        df_clean, active_cars = clean_and_preprocess_case(df)
        _, df_feat_z = extract_features(df_clean, active_cars)
        train_feat_z_dict[fname] = df_feat_z
        
    # Build pairwise dataset across all 6 cases
    feature_keys = ["mean_rel_diff_cooling_z", "mean_t_minus_set_cooling_z", "p95_rel_diff_z", "frac_above_median_z", "persistence_15m_z"]
    pair_rows = []
    
    for fname, df_feat in train_feat_z_dict.items():
        true_car = label_map[fname]
        cars = df_feat.index.tolist()
        for ca in cars:
            for cb in cars:
                if ca == cb:
                    continue
                diff_dict = {f"diff_{k}": df_feat.loc[ca, k] - df_feat.loc[cb, k] for k in feature_keys}
                if ca == true_car:
                    target = 1
                elif cb == true_car:
                    target = 0
                else:
                    target = 0.5
                diff_dict["target"] = target
                pair_rows.append(diff_dict)
                
    df_pairs = pd.DataFrame(pair_rows)
    decisive = df_pairs[df_pairs["target"].isin([0, 1])]
    diff_cols = [c for c in decisive.columns if c.startswith("diff_")]
    X_train = decisive[diff_cols]
    y_train = decisive["target"].astype(int)
    
    # Fit production regularized pairwise model
    pairwise_model = LogisticRegression(C=0.2, penalty='l2', solver='lbfgs', random_state=42)
    pairwise_model.fit(X_train, y_train)
    print(f"Fitted Pairwise Logistic Regression on {len(X_train)} decisive pairwise instances.")
    
    # Assemble model bundle
    bundle = {
        "model_type": "Hybrid_Health_Index_Ensemble",
        "subsystem": "ACV",
        "version": "1.0.0",
        "pairwise_model": pairwise_model,
        "feature_keys": feature_keys,
        "diff_cols": diff_cols,
        "physics_weights": {
            "w1_rel_diff": 1.0,
            "w2_setpoint_error": 0.5,
            "w3_p95_delta": 0.25,
            "w4_frac_above_median": 0.25,
            "w5_persistence": 0.25
        },
        "ensemble_weights": {
            "physics_weight": 0.75,
            "pairwise_linear_weight": 0.25
        },
        "cleaning_thresholds": {
            "temp_min": 5.0,
            "temp_max": 45.0,
            "rolling_window_steps": 30,
            "elevation_threshold_deg": 0.20
        },
        "description": "Rail ACV Refrigerant Leakage Diagnosis & Vehicle Ranking Engine"
    }
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(ACV_DIR, exist_ok=True)
    
    dest_paths = [
        os.path.join(MODELS_DIR, "acv_model_bundle.joblib"),
        os.path.join(ACV_DIR, "acv_model_bundle.joblib")
    ]
    
    for p in dest_paths:
        joblib.dump(bundle, p)
        print(f"Saved model bundle ({os.path.getsize(p)/1024:.1f} KB) to: {p}")

if __name__ == "__main__":
    main()
