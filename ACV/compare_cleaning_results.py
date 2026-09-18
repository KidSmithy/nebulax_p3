"""
ACV Data Cleaning Benchmark & Before-vs-After Comparative Evaluation
-------------------------------------------------------------------
Compares:
1. Data Hygiene (row counts, dropout zeros, non-physical values, depot shutdowns).
2. Signal Stability (temperature statistics, variance, step discontinuities).
3. Modeling Performance (Leave-One-Case-Out CV rank-decay score, test predictions).
"""

import os
import sys
import glob
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = r"c:\Users\Rald999\Documents\GitHub\nebulax_p3\PS3"
ACV_DIR = os.path.join(BASE_DIR, "02_Datasets", "ACV")
TRAIN_DIR = os.path.join(ACV_DIR, "Train")
TEST_DIR = os.path.join(ACV_DIR, "Test")

def compute_linear_rank_decay_score(true_faulty_car, ranked_cars):
    true_str = str(true_faulty_car).zfill(2)
    ranked_str = [str(c).zfill(2) for c in ranked_cars]
    n = len(ranked_str)
    if true_str not in ranked_str:
        return 0.0
    r = ranked_str.index(true_str) + 1
    return (n - (r - 1)) / n

def standardize_headers(df):
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
    car_cols = [c for c in df.columns if c.startswith("Car ") and c != "Car model"]
    all_cids = sorted(list(set(c.split(" - ")[0].replace("Car ", "").strip() for c in car_cols)))
    active = []
    for cid in all_cids:
        t_col = f"Car {cid} - Indoor Average Temperature"
        if t_col in df.columns and df[t_col].notnull().sum() > 0:
            active.append(cid)
    return active

# -------------------------------------------------------------
# RAW PIPELINE (Before Cleaning)
# -------------------------------------------------------------
def process_raw(df, active_cars):
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
            
        features[c] = {
            "mean_indoor": float(np.nanmean(tin)),
            "min_indoor": float(np.nanmin(tin)),
            "max_indoor": float(np.nanmax(tin)),
            "mean_rel_diff_cooling": float(np.nanmean(rel_diff[is_cooling])) if len(rel_diff[is_cooling].dropna()) > 0 else float(np.nanmean(rel_diff)),
            "mean_t_minus_set_cooling": float(np.nanmean(t_minus_set[is_cooling])) if len(t_minus_set[is_cooling].dropna()) > 0 else 0.0,
            "zeros_count": int((tin == 0.0).sum()),
            "step_jumps": int((tin.diff().abs() > 5.0).sum())
        }
    df_feat = pd.DataFrame(features).T
    composite_score = df_feat['mean_rel_diff_cooling'] + 0.5 * df_feat['mean_t_minus_set_cooling']
    ranked_cars = composite_score.sort_values(ascending=False).index.tolist()
    return df_feat, ranked_cars

# -------------------------------------------------------------
# CLEANED PIPELINE (After Cleaning)
# -------------------------------------------------------------
def clean_dataset(df, active_cars):
    df_clean = df.copy()
    
    # 1. Clean sensor dropouts & physical impossibilities (<= 5°C or >= 45°C)
    for cid in active_cars:
        t_col = f"Car {cid} - Indoor Average Temperature"
        if t_col in df_clean.columns:
            s = pd.to_numeric(df_clean[t_col], errors='coerce')
            # Mask out non-physical dropouts (e.g. 0.0°C CAN bus glitch)
            s = s.mask((s <= 5.0) | (s >= 45.0), np.nan).ffill().bfill()
            df_clean[t_col] = s
            
        # Mask out invalid telemetry flags
        valid_col = f"Car {cid} - ACV Information Valid"
        if valid_col in df_clean.columns:
            invalid_mask = df_clean[valid_col].astype(str).str.contains("Invalid", na=False)
            df_clean.loc[invalid_mask, t_col] = np.nan
            
    # 2. Drop complete depot shutdown rows where all active cars are NaN
    active_temp_cols = [f"Car {cid} - Indoor Average Temperature" for cid in active_cars]
    df_clean = df_clean.dropna(subset=active_temp_cols, how='all').reset_index(drop=True)
    
    return df_clean

def process_cleaned(df_clean, active_cars):
    indoor_dict = {}
    cooling_set_dict = {}
    for c in active_cars:
        tin_col = f"Car {c} - Indoor Average Temperature"
        tset_col = f"Car {c} - ACV Control Temperature (Cooling)"
        if tin_col in df_clean.columns:
            indoor_dict[c] = df_clean[tin_col]
        if tset_col in df_clean.columns:
            cooling_set_dict[c] = pd.to_numeric(df_clean[tset_col], errors='coerce')
            
    indoor_df = pd.DataFrame(indoor_dict)
    cooling_set_df = pd.DataFrame(cooling_set_dict)
    median_indoor = indoor_df.median(axis=1)
    
    features = {}
    for c in active_cars:
        tin = indoor_df[c]
        rel_diff = tin - median_indoor
        tset = cooling_set_df[c] if c in cooling_set_df.columns else pd.Series(np.nan, index=df_clean.index)
        t_minus_set = tin - tset
        
        mode_col = f"Car {c} - ACV Running Mode"
        if mode_col in df_clean.columns:
            is_cooling = df_clean[mode_col].astype(str).str.contains("Cooling", na=False)
        else:
            is_cooling = pd.Series(True, index=df_clean.index)
            
        features[c] = {
            "mean_indoor": float(np.nanmean(tin)),
            "min_indoor": float(np.nanmin(tin)),
            "max_indoor": float(np.nanmax(tin)),
            "mean_rel_diff_cooling": float(np.nanmean(rel_diff[is_cooling])) if len(rel_diff[is_cooling].dropna()) > 0 else float(np.nanmean(rel_diff)),
            "mean_t_minus_set_cooling": float(np.nanmean(t_minus_set[is_cooling])) if len(t_minus_set[is_cooling].dropna()) > 0 else 0.0,
            "zeros_count": int((tin == 0.0).sum()),
            "step_jumps": int((tin.diff().abs() > 5.0).sum())
        }
    df_feat = pd.DataFrame(features).T
    composite_score = df_feat['mean_rel_diff_cooling'] + 0.5 * df_feat['mean_t_minus_set_cooling']
    ranked_cars = composite_score.sort_values(ascending=False).index.tolist()
    return df_feat, ranked_cars

# -------------------------------------------------------------
# RUN FULL COMPARISON
# -------------------------------------------------------------
def run_comparison():
    print("=" * 90)
    print(" ACV DATA CLEANING BENCHMARK: BEFORE VS AFTER COMPARISON ")
    print("=" * 90)
    
    labels_file = os.path.join(ACV_DIR, "Train_Labels.csv")
    df_labels = pd.read_csv(labels_file)
    label_map = {row['filename']: str(row['faulty_car']).zfill(2) for _, row in df_labels.iterrows()}
    
    train_files = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.xlsx")))
    test_files = glob.glob(os.path.join(TEST_DIR, "*.xlsx"))
    all_files = [("Train", f) for f in train_files] + [("Test", f) for f in test_files]
    
    file_hygiene_comparison = []
    cv_comparison = []
    
    test_features_before = None
    test_features_after = None
    test_rank_before = None
    test_rank_after = None
    
    for split, fpath in all_files:
        fname = os.path.basename(fpath)
        true_car = label_map.get(fname, None)
        
        df_raw = pd.read_excel(fpath)
        df_raw = standardize_headers(df_raw)
        active_cars = detect_active_cars(df_raw)
        
        # Raw process
        feat_raw, rank_raw = process_raw(df_raw, active_cars)
        
        # Clean process
        df_clean = clean_dataset(df_raw, active_cars)
        feat_clean, rank_clean = process_cleaned(df_clean, active_cars)
        
        # Hygiene record
        raw_zeros = feat_raw['zeros_count'].sum()
        clean_zeros = feat_clean['zeros_count'].sum()
        raw_jumps = feat_raw['step_jumps'].sum()
        clean_jumps = feat_clean['step_jumps'].sum()
        
        file_hygiene_comparison.append({
            "File": fname,
            "Raw Rows": len(df_raw),
            "Clean Rows": len(df_clean),
            "Rows Removed (Depot Off)": len(df_raw) - len(df_clean),
            "Raw Zeros (0°C)": raw_zeros,
            "Clean Zeros": clean_zeros,
            "Raw Step Jumps (>5°C)": raw_jumps,
            "Clean Step Jumps": clean_jumps
        })
        
        if split == "Train":
            score_raw = compute_linear_rank_decay_score(true_car, rank_raw)
            score_clean = compute_linear_rank_decay_score(true_car, rank_clean)
            pos_raw = rank_raw.index(true_car) + 1
            pos_clean = rank_clean.index(true_car) + 1
            
            cv_comparison.append({
                "Case": fname,
                "Fault Car": true_car,
                "Raw Rank": f"{pos_raw}/{len(active_cars)}",
                "Raw Score": f"{score_raw:.4f}",
                "Clean Rank": f"{pos_clean}/{len(active_cars)}",
                "Clean Score": f"{score_clean:.4f}",
                "Rank Raw Order": "|".join(rank_raw),
                "Rank Clean Order": "|".join(rank_clean)
            })
        else:
            test_features_before = feat_raw
            test_features_after = feat_clean
            test_rank_before = rank_raw
            test_rank_after = rank_clean
            
    # Display 1: Data Hygiene Table
    print("\n" + "=" * 90)
    print("1. DATA HYGIENE & NOISE FILTERING COMPARISON")
    print("=" * 90)
    df_hyg = pd.DataFrame(file_hygiene_comparison)
    print(df_hyg.to_string(index=False))
    
    # Display 2: Train CV Evaluation Comparison
    print("\n" + "=" * 90)
    print("2. MODELING PERFORMANCE COMPARISON ON TRAINING CASES (LEAVE-ONE-CASE-OUT CV)")
    print("=" * 90)
    df_cv_comp = pd.DataFrame(cv_comparison)
    print(df_cv_comp[['Case', 'Fault Car', 'Raw Rank', 'Raw Score', 'Clean Rank', 'Clean Score']].to_string(index=False))
    
    raw_mean_score = np.mean([float(r['Raw Score']) for r in cv_comparison])
    clean_mean_score = np.mean([float(r['Clean Score']) for r in cv_comparison])
    print(f"\n>> Mean CV Score BEFORE Cleaning: {raw_mean_score:.4f} / 1.0000 <<")
    print(f">> Mean CV Score AFTER Cleaning : {clean_mean_score:.4f} / 1.0000 <<")
    
    # Display 3: Test Case Detailed Comparison
    print("\n" + "=" * 90)
    print("3. TEST CASE TELEMETRY & PREDICTION COMPARISON (acv_test_case.xlsx)")
    print("=" * 90)
    print(f"Predicted Ranking BEFORE Cleaning: {'|'.join(test_rank_before)}")
    print(f"Predicted Ranking AFTER Cleaning : {'|'.join(test_rank_after)}")
    
    # Compare Car 04 specifically (the car that had 19 zeros!)
    print("\n--- Impact on Car 04 (Which had 19 digital dropout zeros) ---")
    c4_raw = test_features_before.loc['04']
    c4_clean = test_features_after.loc['04']
    print(f"  BEFORE: Min Temp = {c4_raw['min_indoor']:.1f}°C | Zeros = {c4_raw['zeros_count']} | Step Jumps = {c4_raw['step_jumps']}")
    print(f"  AFTER : Min Temp = {c4_clean['min_indoor']:.1f}°C | Zeros = {c4_clean['zeros_count']} | Step Jumps = {c4_clean['step_jumps']}")
    
    # Full Test Car Side-by-Side Comparison
    print("\n--- Side-by-Side Comparison of Test Cars ---")
    test_comp = []
    for c in test_rank_after:
        test_comp.append({
            "Car": c,
            "Raw Mean T": f"{test_features_before.loc[c, 'mean_indoor']:.3f}",
            "Clean Mean T": f"{test_features_after.loc[c, 'mean_indoor']:.3f}",
            "Raw Min T": f"{test_features_before.loc[c, 'min_indoor']:.1f}",
            "Clean Min T": f"{test_features_after.loc[c, 'min_indoor']:.1f}",
            "Raw Rel Diff": f"{test_features_before.loc[c, 'mean_rel_diff_cooling']:.4f}",
            "Clean Rel Diff": f"{test_features_after.loc[c, 'mean_rel_diff_cooling']:.4f}",
            "Raw Zeros": int(test_features_before.loc[c, 'zeros_count']),
            "Clean Zeros": int(test_features_after.loc[c, 'zeros_count'])
        })
    df_test_comp = pd.DataFrame(test_comp)
    print(df_test_comp.to_string(index=False))

if __name__ == "__main__":
    run_comparison()
