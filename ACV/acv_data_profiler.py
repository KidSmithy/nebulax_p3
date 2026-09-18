"""
========================================================================================
Train ACV (Air Conditioning & Ventilation) Refrigerant Leakage Profiler & Benchmark
========================================================================================
Author: AI Pair Programmer (Antigravity)
Purpose: Extracts exact telemetry signatures, cross-car differential statistics,
         handles schema differences across diverse consist layouts, and evaluates
         predictive ranking models using the official Linear Rank Decay metric.

Can be run standalone to generate statistical summaries or supplement Manus AI models:
    python acv_data_profiler.py
========================================================================================
"""

import os
import sys
import json
import glob
import pandas as pd
import numpy as np

# Ensure UTF-8 console output for multilingual headers / sheet names
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ACV_DIR = os.path.join(BASE_DIR, "02_Datasets", "ACV")
TRAIN_DIR = os.path.join(ACV_DIR, "Train")
TEST_DIR = os.path.join(ACV_DIR, "Test")

def compute_linear_rank_decay_score(true_faulty_car, ranked_cars):
    """
    Computes official ACV evaluation metric:
        Score = (n - (r - 1)) / n
    where:
        n = number of cars in the train consist
        r = 1-based rank position of the true faulty car
    """
    true_str = str(true_faulty_car).zfill(2)
    ranked_str = [str(c).zfill(2) for c in ranked_cars]
    n = len(ranked_str)
    if true_str not in ranked_str:
        return 0.0
    r = ranked_str.index(true_str) + 1
    return (n - (r - 1)) / n

def standardize_case_columns(df):
    """
    Standardizes schema variations across all case files:
      - Cases 1-3, Test: 'Car XX - Outdoor Average Temperature'
      - Cases 5-6:       'Car XX - Outside Temperature Sensor Reading'
      - Case 4:          'Car XX - Fresh Air Temperature Detected Value',
                         'Car XX - Passenger Cabin Temperature Detected Value',
                         'Car XX - Target Temperature Value'
    """
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
    return df.rename(columns=rename_dict)

def detect_active_cars(df):
    """
    Identifies active cars (handling variable consist lengths such as Case 04's 4-car consist).
    """
    car_cols = [c for c in df.columns if c.startswith("Car ") and c != "Car model"]
    all_cids = sorted(list(set(c.split(" - ")[0].replace("Car ", "").strip() for c in car_cols)))
    
    active = []
    for cid in all_cids:
        # Check indoor temperature column
        t_col = f"Car {cid} - Indoor Average Temperature"
        if t_col in df.columns:
            if df[t_col].notnull().sum() > 0:
                active.append(cid)
        else:
            # Fallback to any car column
            matching = [c for c in df.columns if c.startswith(f"Car {cid} - ")]
            if matching and df[matching[0]].notnull().sum() > 0:
                active.append(cid)
    return active

def extract_cross_car_features(df, active_cars):
    """
    Extracts cross-car relative temperature differentials and setpoint tracking errors.
    By subtracting the consist-wide median at every timestamp, external ambient temperature,
    weather dynamics, diurnal cycles, and line-wide heating loads are neutralized.
    """
    indoor_dict = {}
    cooling_set_dict = {}
    
    for c in active_cars:
        tin_col = f"Car {c} - Indoor Average Temperature"
        tset_col = f"Car {c} - ACV Control Temperature (Cooling)"
        if tin_col in df.columns:
            indoor_dict[c] = pd.to_numeric(df[tin_col], errors='coerce')
        if tset_col in df.columns:
            cooling_set_dict[c] = pd.to_numeric(df[tset_col], errors='coerce')
            
    indoor_df = pd.DataFrame(indoor_dict)
    cooling_set_df = pd.DataFrame(cooling_set_dict)
    
    # Consist baseline (median filters out single-car outliers)
    median_indoor = indoor_df.median(axis=1)
    mean_indoor = indoor_df.mean(axis=1)
    
    features = {}
    for c in active_cars:
        tin = indoor_df[c]
        rel_diff = tin - median_indoor
        
        tset = cooling_set_df[c] if c in cooling_set_df.columns else pd.Series(np.nan, index=df.index)
        t_minus_set = tin - tset
        
        # Check if cooling mode is active
        mode_col = f"Car {c} - ACV Running Mode"
        if mode_col in df.columns:
            is_cooling = df[mode_col].astype(str).str.contains("Cooling", na=False)
        else:
            is_cooling = pd.Series(True, index=df.index)
            
        rel_cooling = rel_diff[is_cooling]
        t_minus_set_cooling = t_minus_set[is_cooling]
        
        feat = {
            "mean_indoor_temp": float(np.nanmean(tin)),
            "std_indoor_temp": float(np.nanstd(tin)),
            "max_indoor_temp": float(np.nanmax(tin)),
            "mean_rel_diff": float(np.nanmean(rel_diff)),
            "median_rel_diff": float(np.nanmedian(rel_diff)),
            "p75_rel_diff": float(np.nanpercentile(rel_diff.dropna(), 75)) if len(rel_diff.dropna()) > 0 else 0.0,
            "p90_rel_diff": float(np.nanpercentile(rel_diff.dropna(), 90)) if len(rel_diff.dropna()) > 0 else 0.0,
            "p95_rel_diff": float(np.nanpercentile(rel_diff.dropna(), 95)) if len(rel_diff.dropna()) > 0 else 0.0,
            "max_rel_diff": float(np.nanmax(rel_diff)),
            "mean_rel_diff_cooling": float(np.nanmean(rel_cooling)) if len(rel_cooling.dropna()) > 0 else float(np.nanmean(rel_diff)),
            "mean_t_minus_set_cooling": float(np.nanmean(t_minus_set_cooling)) if len(t_minus_set_cooling.dropna()) > 0 else 0.0,
            "pct_above_median": float((rel_diff > 0.2).mean() * 100),
            "integrated_positive_deficit": float(np.nansum(np.maximum(0, t_minus_set_cooling)))
        }
        features[c] = feat
        
    return pd.DataFrame(features).T

def run_pipeline():
    print("=" * 80)
    print(" TRAIN ACV SUBSYSTEM DEEP PROFILER & BENCHMARK ")
    print("=" * 80)
    
    # 1. Read ground truth labels
    labels_file = os.path.join(ACV_DIR, "Train_Labels.csv")
    df_labels = pd.read_csv(labels_file)
    label_map = {row['filename']: str(row['faulty_car']).zfill(2) for _, row in df_labels.iterrows()}
    
    print("\n[Ground Truth Training Labels]")
    print(df_labels.to_string(index=False))
    
    # 2. Iterate through training cases
    train_files = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.xlsx")))
    cv_records = []
    
    print("\n" + "=" * 80)
    print(" LEAVE-ONE-CASE-OUT EVALUATION ON TRAINING CASES ")
    print("=" * 80)
    
    for fpath in train_files:
        fname = os.path.basename(fpath)
        true_car = label_map[fname]
        
        df = pd.read_excel(fpath)
        df = standardize_case_columns(df)
        active_cars = detect_active_cars(df)
        df_feat = extract_cross_car_features(df, active_cars)
        
        # Model Ranking Function:
        # Score = Mean Relative Temperature Difference during Cooling + 0.5 * Setpoint Tracking Error
        composite_anomaly = df_feat['mean_rel_diff_cooling'] + 0.5 * df_feat['mean_t_minus_set_cooling']
        ranked_cars = composite_anomaly.sort_values(ascending=False).index.tolist()
        
        score = compute_linear_rank_decay_score(true_car, ranked_cars)
        rank = ranked_cars.index(true_car) + 1
        
        cv_records.append({
            "File": fname,
            "Active Cars": len(active_cars),
            "True Fault": true_car,
            "Predicted Rank 1": ranked_cars[0],
            "Fault Rank": f"{rank}/{len(active_cars)}",
            "Decay Score": f"{score:.4f}",
            "Complete Ranking": "|".join(ranked_cars)
        })
        
    df_cv = pd.DataFrame(cv_records)
    print(df_cv[['File', 'Active Cars', 'True Fault', 'Predicted Rank 1', 'Fault Rank', 'Decay Score']].to_string(index=False))
    
    scores = [float(r['Decay Score']) for r in cv_records]
    print(f"\n>> Mean CV Linear Rank Decay Score: {np.mean(scores):.4f} (Max: 1.0000) <<")
    
    # 3. Test Case Inference
    print("\n" + "=" * 80)
    print(" TEST SET INFERENCE: acv_test_case.xlsx ")
    print("=" * 80)
    test_files = glob.glob(os.path.join(TEST_DIR, "*.xlsx"))
    if test_files:
        test_fpath = test_files[0]
        test_fname = os.path.basename(test_fpath)
        df_test = pd.read_excel(test_fpath)
        df_test = standardize_case_columns(df_test)
        test_active_cars = detect_active_cars(df_test)
        df_test_feat = extract_cross_car_features(df_test, test_active_cars)
        
        test_anomaly = df_test_feat['mean_rel_diff_cooling'] + 0.5 * df_test_feat['mean_t_minus_set_cooling']
        test_ranking = test_anomaly.sort_values(ascending=False).index.tolist()
        test_ranked_str = "|".join(test_ranking)
        
        print(f"Target File: {test_fname}")
        print(f"Predicted Full Ranking: {test_ranked_str}")
        print(f"Top 3 Fault Suspects: 1st -> Car {test_ranking[0]}, 2nd -> Car {test_ranking[1]}, 3rd -> Car {test_ranking[2]}")
        
        print("\nFeature Summary Table for Test Consist:")
        view_cols = ['mean_indoor_temp', 'mean_rel_diff', 'mean_rel_diff_cooling', 'mean_t_minus_set_cooling', 'p95_rel_diff', 'pct_above_median']
        print(df_test_feat[view_cols].sort_values(by='mean_rel_diff_cooling', ascending=False).to_string())
        
        # Save submission file
        pred_csv = os.path.join(BASE_DIR, "04_Example_Submission", "acv_predictions.csv")
        out_df = pd.DataFrame([{"file_id": test_fname, "ranked_cars": test_ranked_str}])
        out_path = os.path.join(BASE_DIR, "acv_predictions.csv")
        out_df.to_csv(out_path, index=False)
        print(f"\nGenerated submission CSV: {out_path}")
        print(f"Schema verification:\n{out_df.to_string(index=False)}")

if __name__ == "__main__":
    run_pipeline()
