"""
ACV Dataset Profiler and Feature Engineering Pipeline
-----------------------------------------------------
Extracts comprehensive telemetry statistics, cross-car differentials,
evaluates rank-decay scoring across all 6 training cases, and produces
reproducible data summaries for model selection in Manus AI.
"""

import os
import sys
import json
import glob
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = r"c:\Users\Rald999\Documents\GitHub\nebulax_p3\PS3"
ACV_DIR = os.path.join(BASE_DIR, "02_Datasets", "ACV")

def linear_rank_decay_score(true_faulty_car, ranked_cars):
    """
    Computes official ACV evaluation score:
    score = (n - (r - 1)) / n
    where r is 1-based rank of true_faulty_car, n is number of cars.
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
    Standardizes column names across diverse ACV case schemas.
    """
    rename_dict = {}
    for col in df.columns:
        # Standardize outdoor temp
        if "Outside Temperature Sensor Reading" in col:
            rename_dict[col] = col.replace("Outside Temperature Sensor Reading", "Outdoor Average Temperature")
        elif "Fresh Air Temperature Detected Value" in col:
            rename_dict[col] = col.replace("Fresh Air Temperature Detected Value", "Outdoor Average Temperature")
        # Standardize indoor temp in case 04
        elif "Passenger Cabin Temperature Detected Value" in col:
            rename_dict[col] = col.replace("Passenger Cabin Temperature Detected Value", "Indoor Average Temperature")
        # Standardize target temp in case 04
        elif "Target Temperature Value" in col:
            rename_dict[col] = col.replace("Target Temperature Value", "ACV Control Temperature (Cooling)")
    return df.rename(columns=rename_dict)

def get_active_cars(df):
    """
    Identifies cars that have valid (non-all-NaN) data.
    """
    car_cols = [c for c in df.columns if c.startswith("Car ") and c != "Car model"]
    all_car_ids = sorted(list(set(c.split(" - ")[0].replace("Car ", "").strip() for c in car_cols)))
    
    active_cars = []
    for cid in all_car_ids:
        # Check indoor temp or any car col
        matching_cols = [c for c in df.columns if c.startswith(f"Car {cid} - ")]
        if matching_cols:
            valid_sum = df[matching_cols[0]].notnull().sum()
            if valid_sum > 0:
                active_cars.append(cid)
    return active_cars

def extract_car_features(df, active_cars):
    """
    Extracts cross-car differential and thermal deficit features for each car.
    """
    # 1. Indoor temp matrix
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
    
    # Train median indoor temp at each timestamp (filters out ambient/weather noise)
    median_indoor = indoor_df.median(axis=1)
    mean_indoor = indoor_df.mean(axis=1)
    
    features = {}
    for c in active_cars:
        tin = indoor_df[c]
        rel_diff = tin - median_indoor
        
        # Temperature deficit vs setpoint
        tset = cooling_set_df[c] if c in cooling_set_df.columns else pd.Series(np.nan, index=df.index)
        t_minus_set = tin - tset
        
        # Cooling active mask
        mode_col = f"Car {c} - ACV Running Mode"
        if mode_col in df.columns:
            is_cooling = df[mode_col].astype(str).str.contains("Cooling", na=False)
        else:
            is_cooling = pd.Series(True, index=df.index)
            
        rel_diff_cooling = rel_diff[is_cooling]
        t_minus_set_cooling = t_minus_set[is_cooling]
        
        # Thermal inertia / persistence of positive error (failing to cool)
        pos_error = np.maximum(0, t_minus_set_cooling)
        
        feat = {
            "mean_indoor_temp": float(np.nanmean(tin)),
            "max_indoor_temp": float(np.nanmax(tin)),
            "std_indoor_temp": float(np.nanstd(tin)),
            "mean_rel_diff": float(np.nanmean(rel_diff)),
            "median_rel_diff": float(np.nanmedian(rel_diff)),
            "p75_rel_diff": float(np.nanpercentile(rel_diff.dropna(), 75)) if len(rel_diff.dropna()) > 0 else 0.0,
            "p90_rel_diff": float(np.nanpercentile(rel_diff.dropna(), 90)) if len(rel_diff.dropna()) > 0 else 0.0,
            "p95_rel_diff": float(np.nanpercentile(rel_diff.dropna(), 95)) if len(rel_diff.dropna()) > 0 else 0.0,
            "mean_rel_diff_cooling": float(np.nanmean(rel_diff_cooling)) if len(rel_diff_cooling.dropna()) > 0 else float(np.nanmean(rel_diff)),
            "mean_t_minus_set_cooling": float(np.nanmean(t_minus_set_cooling)) if len(t_minus_set_cooling.dropna()) > 0 else 0.0,
            "integrated_positive_error": float(np.nansum(pos_error)),
            "frac_above_train_median": float((rel_diff > 0.25).mean()),
            "frac_above_setpoint": float((t_minus_set > 0.5).mean())
        }
        features[c] = feat
        
    return pd.DataFrame(features).T

def main():
    print("="*80)
    print("ACV COMPREHENSIVE PROFILING & BENCHMARK")
    print("="*80)
    
    # Load Train Labels
    df_labels = pd.read_csv(os.path.join(ACV_DIR, "Train_Labels.csv"))
    print(df_labels)
    
    label_map = {row['filename']: str(row['faulty_car']).zfill(2) for _, row in df_labels.iterrows()}
    
    train_files = sorted(glob.glob(os.path.join(ACV_DIR, "Train", "*.xlsx")))
    test_files = sorted(glob.glob(os.path.join(ACV_DIR, "Test", "*.xlsx")))
    
    results = []
    
    for fpath in train_files:
        fname = os.path.basename(fpath)
        true_fault = label_map.get(fname)
        df = pd.read_excel(fpath)
        df = standardize_case_columns(df)
        active_cars = get_active_cars(df)
        df_feat = extract_car_features(df, active_cars)
        
        # Rank cars by thermal deficit (Composite score)
        # Composite score: mean_rel_diff_cooling + 0.5 * mean_t_minus_set_cooling
        composite_score = df_feat['mean_rel_diff_cooling'] + 0.5 * df_feat['mean_t_minus_set_cooling']
        ranked_cars = composite_score.sort_values(ascending=False).index.tolist()
        
        score = linear_rank_decay_score(true_fault, ranked_cars)
        rank = ranked_cars.index(true_fault) + 1
        
        results.append({
            "filename": fname,
            "true_fault": true_fault,
            "cars_count": len(active_cars),
            "predicted_rank_1": ranked_cars[0],
            "true_rank": rank,
            "score": score,
            "full_ranking": "|".join(ranked_cars)
        })
        print(f"[{fname}] True Fault: Car {true_fault} | Ranked: {rank}/{len(active_cars)} | Score: {score:.4f} | Order: {'|'.join(ranked_cars)}")

    df_res = pd.DataFrame(results)
    print("\n" + "="*80)
    print("OVERALL LEAVE-ONE-OUT BENCHMARK SUMMARY (TRAIN CASES)")
    print("="*80)
    print(df_res[['filename', 'true_fault', 'true_rank', 'score', 'full_ranking']].to_string(index=False))
    print(f"\nAverage ACV Linear Rank Decay Score: {df_res['score'].mean():.4f} (Max possible: 1.0000)")
    
    # Run on Test Case
    print("\n" + "="*80)
    print("TEST CASE INFERENCE & PREDICTION GENERATION")
    print("="*80)
    test_fpath = test_files[0]
    test_fname = os.path.basename(test_fpath)
    df_test = pd.read_excel(test_fpath)
    df_test = standardize_case_columns(df_test)
    test_cars = get_active_cars(df_test)
    df_test_feat = extract_car_features(df_test, test_cars)
    
    test_composite = df_test_feat['mean_rel_diff_cooling'] + 0.5 * df_test_feat['mean_t_minus_set_cooling']
    test_ranked = test_composite.sort_values(ascending=False).index.tolist()
    ranked_str = "|".join(test_ranked)
    
    print(f"File: {test_fname}")
    print(f"Predicted Ranked Cars: {ranked_str}")
    print("\nCar Telemetry Feature Ranking Table:")
    print(df_test_feat[['mean_indoor_temp', 'mean_rel_diff', 'mean_rel_diff_cooling', 'mean_t_minus_set_cooling', 'p95_rel_diff']].sort_values(by='mean_rel_diff_cooling', ascending=False))
    
    # Save predictions in submission format
    sub_df = pd.DataFrame([{"file_id": test_fname, "ranked_cars": ranked_str}])
    sub_out_path = os.path.join(BASE_DIR, "acv_baseline_predictions.csv")
    sub_df.to_csv(sub_out_path, index=False)
    print(f"\nSaved benchmark prediction to: {sub_out_path}")

if __name__ == "__main__":
    main()
