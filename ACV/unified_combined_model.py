"""
Unified Supervised Model: Combine & Concat All Training Cases
-------------------------------------------------------------
Demonstrates how all 6 training cases are combined/concatenated into
a single master training dataset, fitting ONE single model, and using
that single model to predict on any held-out test file.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = r"c:\Users\Rald999\Documents\GitHub\nebulax_p3\PS3"
ACV_DIR = os.path.join(BASE_DIR, "02_Datasets", "ACV")
TRAIN_DIR = os.path.join(ACV_DIR, "Train")
TEST_DIR = os.path.join(ACV_DIR, "Test")

def clean_and_normalize(df):
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
    
    # Active cars
    car_cols = [c for c in df.columns if c.startswith("Car ") and c != "Car model"]
    all_cids = sorted(list(set(c.split(" - ")[0].replace("Car ", "").strip() for c in car_cols)))
    active_cars = [cid for cid in all_cids if f"Car {cid} - Indoor Average Temperature" in df.columns 
                   and df[f"Car {cid} - Indoor Average Temperature"].notnull().sum() > 0]
                   
    # Clean dropouts
    for cid in active_cars:
        t_col = f"Car {cid} - Indoor Average Temperature"
        s = pd.to_numeric(df[t_col], errors='coerce')
        s = s.mask((s <= 5.0) | (s >= 45.0), np.nan).ffill().bfill()
        df[t_col] = s
        
    return df, active_cars

def extract_car_level_features(df, active_cars, case_name, is_train=True, true_faulty_car=None):
    """
    Extracts high-dimensional tabular features for each car in a given case.
    """
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
    
    records = []
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
        
        rec = {
            "case_id": case_name,
            "car_id": c,
            "mean_indoor": float(np.nanmean(tin)),
            "std_indoor": float(np.nanstd(tin)),
            "mean_rel_diff": float(np.nanmean(rel_diff)),
            "p75_rel_diff": float(np.nanpercentile(rel_diff.dropna(), 75)) if len(rel_diff.dropna()) > 0 else 0.0,
            "p90_rel_diff": float(np.nanpercentile(rel_diff.dropna(), 90)) if len(rel_diff.dropna()) > 0 else 0.0,
            "p95_rel_diff": float(np.nanpercentile(rel_diff.dropna(), 95)) if len(rel_diff.dropna()) > 0 else 0.0,
            "max_rel_diff": float(np.nanmax(rel_diff)),
            "mean_rel_diff_cooling": float(np.nanmean(rel_cooling)) if len(rel_cooling.dropna()) > 0 else float(np.nanmean(rel_diff)),
            "mean_t_minus_set_cooling": float(np.nanmean(t_minus_set_cooling)) if len(t_minus_set_cooling.dropna()) > 0 else 0.0,
            "p90_t_minus_set_cooling": float(np.nanpercentile(t_minus_set_cooling.dropna(), 90)) if len(t_minus_set_cooling.dropna()) > 0 else 0.0,
            "pct_above_median": float((rel_diff > 0.2).mean() * 100),
            "integrated_positive_deficit": float(np.nansum(np.maximum(0, t_minus_set_cooling)))
        }
        
        if is_train:
            rec["is_faulty"] = 1 if c == true_faulty_car else 0
            
        records.append(rec)
        
    return pd.DataFrame(records)

def main():
    print("="*85)
    print("DEMONSTRATION: COMBINE & CONCAT ALL TRAINING CASES INTO ONE UNIFIED DATASET")
    print("="*85)
    
    # 1. Load Ground Truth Labels
    labels_file = os.path.join(ACV_DIR, "Train_Labels.csv")
    df_labels = pd.read_csv(labels_file)
    label_map = {row['filename']: str(row['faulty_car']).zfill(2) for _, row in df_labels.iterrows()}
    
    train_files = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.xlsx")))
    
    # 2. Extract and Concatenate All Training Cases Together
    all_train_records = []
    print("\nExtracting features from each training case...")
    for fpath in train_files:
        fname = os.path.basename(fpath)
        true_car = label_map[fname]
        df = pd.read_excel(fpath)
        df_clean, active_cars = clean_and_normalize(df)
        df_case_features = extract_car_level_features(df_clean, active_cars, fname, is_train=True, true_faulty_car=true_car)
        all_train_records.append(df_case_features)
        print(f"  + {fname}: extracted {len(df_case_features)} cars (Faulty: Car {true_car})")
        
    # CONCATENATE TOGETHER!
    master_train_df = pd.concat(all_train_records, axis=0, ignore_index=True)
    
    print("\n" + "="*85)
    print("MASTER CONCATENATED TRAINING DATASET (ALL 6 CASES COMBINED TOGETHER):")
    print("="*85)
    print(f"Total Combined Training Samples: {len(master_train_df)} cars across 6 train cases")
    print(f"Class Breakdown: Faulty cars = {master_train_df['is_faulty'].sum()}, Healthy cars = {(master_train_df['is_faulty']==0).sum()}")
    print("\nSample Rows from Master Training Table:")
    cols_to_show = ["case_id", "car_id", "mean_rel_diff_cooling", "mean_t_minus_set_cooling", "p95_rel_diff", "is_faulty"]
    print(master_train_df[cols_to_show].head(10).to_string(index=False))
    print("...")
    print(master_train_df[cols_to_show].tail(6).to_string(index=False))
    
    # 3. Train ONE Single Model on the Combined Dataset
    print("\n" + "="*85)
    print("TRAINING ONE SINGLE UNIFIED MODEL ON ALL CONCATENATED DATA:")
    print("="*85)
    
    feature_cols = [
        "mean_rel_diff", "p75_rel_diff", "p90_rel_diff", "p95_rel_diff",
        "max_rel_diff", "mean_rel_diff_cooling", "mean_t_minus_set_cooling",
        "p90_t_minus_set_cooling", "pct_above_median"
    ]
    
    X_train = master_train_df[feature_cols]
    y_train = master_train_df["is_faulty"]
    
    # We train ONE single model (e.g. GradientBoostingClassifier or LogisticRegression)
    model = GradientBoostingClassifier(n_estimators=50, max_depth=2, random_state=42)
    model.fit(X_train, y_train)
    print(f"Trained single {type(model).__name__} on all {len(X_train)} instances.")
    
    # Feature importances
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nModel Learned Feature Importances:")
    for feat, imp in importances.items():
        print(f"  - {feat:<26}: {imp*100:.2f}%")
        
    # 4. Predict on Test Case using the ONE trained model
    print("\n" + "="*85)
    print("PREDICTING ON HELD-OUT TEST FILE USING THE SINGLE TRAINED MODEL:")
    print("="*85)
    test_file = glob.glob(os.path.join(TEST_DIR, "*.xlsx"))[0]
    test_fname = os.path.basename(test_file)
    df_test = pd.read_excel(test_file)
    df_test_clean, test_active = clean_and_normalize(df_test)
    test_df = extract_car_level_features(df_test_clean, test_active, test_fname, is_train=False)
    
    X_test = test_df[feature_cols]
    
    # Predict continuous fault probability / decision score for each car
    fault_probabilities = model.predict_proba(X_test)[:, 1]
    test_df["fault_probability"] = fault_probabilities
    
    # Rank cars by predicted fault probability (highest to lowest)
    test_ranked_df = test_df.sort_values(by="fault_probability", ascending=False)
    ranked_cars = test_ranked_df["car_id"].tolist()
    ranked_str = "|".join(ranked_cars)
    
    print(f"Target Test File: {test_fname}")
    print(f"Predicted Full Ranking: {ranked_str}")
    print("\nSingle Model Output Predictions per Car:")
    print(test_ranked_df[["car_id", "fault_probability", "mean_rel_diff_cooling", "mean_t_minus_set_cooling"]].to_string(index=False))

if __name__ == "__main__":
    main()
