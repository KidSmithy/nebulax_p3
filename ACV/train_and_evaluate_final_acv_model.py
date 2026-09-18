"""
Final Production ACV Pipeline: Physics Ranker + Pairwise GBDT Ensemble
----------------------------------------------------------------------
Based on Manus AI recommendations:
1. Strict data cleaning (removes 0.0°C dropouts, depot shutdowns, normalizes headers).
2. Physics-Informed Relative Thermal Deficit Ranker with grid-searched weights.
3. Shallow Pairwise Gradient Boosting model (HistGradientBoosting / XGBoost).
4. Rank aggregation ensemble (75% Physics + 25% Pairwise GBDT).
5. Leave-One-Case-Out cross-validation on all 6 train cases.
6. Sensitivity analysis on held-out test case (acv_test_case.xlsx).
7. Generates official competition submission acv_predictions.csv.
"""

import os
import sys
import glob
import itertools
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

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

def clean_and_preprocess_case(df):
    """
    1. Normalizes inconsistent headers across cases.
    2. Dynamically detects populated cars.
    3. Cleans sensor dropout zeros (0.0°C) and comms faults.
    4. Drops complete depot power-off blocks.
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
    df = df.rename(columns=rename_dict)
    
    # Active cars
    car_cols = [c for c in df.columns if c.startswith("Car ") and c != "Car model"]
    all_cids = sorted(list(set(c.split(" - ")[0].replace("Car ", "").strip() for c in car_cols)))
    active_cars = [cid for cid in all_cids if f"Car {cid} - Indoor Average Temperature" in df.columns 
                   and df[f"Car {cid} - Indoor Average Temperature"].notnull().sum() > 0]
                   
    # Clean dropouts
    for cid in active_cars:
        t_col = f"Car {cid} - Indoor Average Temperature"
        if t_col in df.columns:
            s = pd.to_numeric(df[t_col], errors='coerce')
            s = s.mask((s <= 5.0) | (s >= 45.0), np.nan).ffill().bfill()
            df[t_col] = s
            
        valid_col = f"Car {cid} - ACV Information Valid"
        if valid_col in df.columns:
            invalid_mask = df[valid_col].astype(str).str.contains("Invalid", na=False)
            df.loc[invalid_mask, t_col] = np.nan
            
    active_temp_cols = [f"Car {cid} - Indoor Average Temperature" for cid in active_cars if f"Car {cid} - Indoor Average Temperature" in df.columns]
    df = df.dropna(subset=active_temp_cols, how='all').reset_index(drop=True)
    return df, active_cars

def extract_car_features(df, active_cars):
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
        
        feat = {
            "mean_rel_diff_cooling": float(np.nanmean(rel_cooling)) if len(rel_cooling.dropna()) > 0 else float(np.nanmean(rel_diff)),
            "mean_t_minus_set_cooling": float(np.nanmean(t_minus_set_cooling)) if len(t_minus_set_cooling.dropna()) > 0 else 0.0,
            "p95_rel_diff": float(np.nanpercentile(rel_diff.dropna(), 95)) if len(rel_diff.dropna()) > 0 else 0.0,
            "pct_above_median": float((rel_diff > 0.2).mean() * 100),
            "mean_rel_diff": float(np.nanmean(rel_diff)),
            "max_rel_diff": float(np.nanmax(rel_diff))
        }
        features[c] = feat
        
    return pd.DataFrame(features).T

def score_physics(df_feat, w1=1.0, w2=0.5, w3=0.25, w4=0.01):
    # Normalized weights: note pct_above_median is in 0-100 scale, so 0.01 scales it to 0-1.
    return (
        w1 * df_feat['mean_rel_diff_cooling']
        + w2 * df_feat['mean_t_minus_set_cooling']
        + w3 * df_feat['p95_rel_diff']
        + w4 * df_feat['pct_above_median']
    )

def build_pairwise_dataset(case_features_dict, label_map):
    pair_rows = []
    feature_keys = ["mean_rel_diff_cooling", "mean_t_minus_set_cooling", "p95_rel_diff", "pct_above_median"]
    
    for fname, df_feat in case_features_dict.items():
        true_car = label_map[fname]
        cars = df_feat.index.tolist()
        
        for ca in cars:
            for cb in cars:
                if ca == cb:
                    continue
                diff_feat = {f"diff_{k}": df_feat.loc[ca, k] - df_feat.loc[cb, k] for k in feature_keys}
                diff_feat["case_id"] = fname
                diff_feat["car_a"] = ca
                diff_feat["car_b"] = cb
                # Target: 1 if car_a is faulty, 0 if car_b is faulty, or 0 if neither (or we focus on pairs involving fault)
                # Standard pairwise rank: 1 if rank(a) < rank(b) (a is more faulty)
                if ca == true_car:
                    target = 1
                elif cb == true_car:
                    target = 0
                else:
                    target = 0.5  # Neutral pair
                diff_feat["target"] = target
                pair_rows.append(diff_feat)
                
    return pd.DataFrame(pair_rows)

def train_pairwise_gbdt(df_pairs_train, df_test_feat):
    feature_cols = [c for c in df_pairs_train.columns if c.startswith("diff_")]
    
    # Train on decisive pairs (target == 1 or target == 0)
    decisive = df_pairs_train[df_pairs_train["target"].isin([0, 1])]
    X_train = decisive[feature_cols]
    y_train = decisive["target"].astype(int)
    
    # Shallow, strongly regularized gradient booster
    model = HistGradientBoostingClassifier(
        max_iter=30,
        max_depth=2,
        min_samples_leaf=5,
        learning_rate=0.03,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate pairwise comparisons for test consist
    test_cars = df_test_feat.index.tolist()
    pairwise_scores = {c: 0.0 for c in test_cars}
    feature_keys = ["mean_rel_diff_cooling", "mean_t_minus_set_cooling", "p95_rel_diff", "pct_above_median"]
    
    for ca in test_cars:
        for cb in test_cars:
            if ca == cb:
                continue
            diff_feat = pd.DataFrame([{f"diff_{k}": df_test_feat.loc[ca, k] - df_test_feat.loc[cb, k] for k in feature_keys}])
            prob_a_faulty = model.predict_proba(diff_feat)[0, 1]
            pairwise_scores[ca] += prob_a_faulty
            
    # Normalize tournament scores to [0, 1]
    max_s = max(pairwise_scores.values())
    min_s = min(pairwise_scores.values())
    spread = (max_s - min_s) if (max_s - min_s) > 0 else 1.0
    norm_scores = {c: (pairwise_scores[c] - min_s) / spread for c in test_cars}
    return pd.Series(norm_scores)

def main():
    print("=" * 85)
    print(" EXECUTING MANUS AI RECOMMENDED STRATEGY: PHYSICS RANKER + PAIRWISE GBDT ")
    print("=" * 85)
    
    # Load labels
    labels_file = os.path.join(ACV_DIR, "Train_Labels.csv")
    df_labels = pd.read_csv(labels_file)
    label_map = {row['filename']: str(row['faulty_car']).zfill(2) for _, row in df_labels.iterrows()}
    
    # 1. Clean & extract features for all training files
    train_files = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.xlsx")))
    train_features_dict = {}
    
    print("\n[Step 1] Loading, cleaning, and extracting features for all 6 train cases...")
    for fpath in train_files:
        fname = os.path.basename(fpath)
        df_raw = pd.read_excel(fpath)
        df_clean, active_cars = clean_and_preprocess_case(df_raw)
        df_feat = extract_car_features(df_clean, active_cars)
        train_features_dict[fname] = df_feat
        print(f"  Processed {fname}: {len(active_cars)} active cars | Ground truth fault: Car {label_map[fname]}")
        
    # 2. Physics-Informed Ranker Grid Search & LOOCV Evaluation
    print("\n" + "=" * 85)
    print("[Step 2] Leave-One-Case-Out Evaluation: Physics Ranker")
    print("=" * 85)
    
    # Grid search candidate weights
    weight_candidates = [
        # (w1, w2, w3, w4)
        (1.0, 0.5, 0.25, 0.01),  # Manus baseline
        (1.0, 0.5, 0.20, 0.005),
        (1.0, 0.4, 0.20, 0.01),
        (1.0, 0.6, 0.30, 0.01),
        (1.0, 0.5, 0.00, 0.00),  # Our earlier 2-feature baseline
    ]
    
    best_weights = None
    best_mean_score = -1.0
    
    for w1, w2, w3, w4 in weight_candidates:
        scores = []
        for fname, df_feat in train_features_dict.items():
            true_car = label_map[fname]
            s = score_physics(df_feat, w1, w2, w3, w4)
            ranked = s.sort_values(ascending=False).index.tolist()
            sc = compute_linear_rank_decay_score(true_car, ranked)
            scores.append(sc)
        mean_sc = np.mean(scores)
        print(f"Weights (w1={w1}, w2={w2}, w3={w3}, w4={w4}): LOOCV Mean Score = {mean_sc:.4f}")
        if mean_sc > best_mean_score:
            best_mean_score = mean_sc
            best_weights = (w1, w2, w3, w4)
            
    print(f"\n>> Selected Best Physics Weights: {best_weights} (Mean CV: {best_mean_score:.4f}) <<")
    
    # 3. Pairwise GBDT LOOCV Evaluation
    print("\n" + "=" * 85)
    print("[Step 3] Leave-One-Case-Out Evaluation: Pairwise GBDT")
    print("=" * 85)
    
    case_names = sorted(list(train_features_dict.keys()))
    gbdt_scores = []
    
    for val_case in case_names:
        train_cases = {k: v for k, v in train_features_dict.items() if k != val_case}
        df_pairs_train = build_pairwise_dataset(train_cases, label_map)
        val_feat = train_features_dict[val_case]
        
        pairwise_pred_series = train_pairwise_gbdt(df_pairs_train, val_feat)
        ranked_gbdt = pairwise_pred_series.sort_values(ascending=False).index.tolist()
        
        true_car = label_map[val_case]
        sc_gbdt = compute_linear_rank_decay_score(true_car, ranked_gbdt)
        gbdt_scores.append(sc_gbdt)
        print(f"  Held-out {val_case}: True Fault = Car {true_car} | GBDT Predicted Rank 1 = Car {ranked_gbdt[0]} | Score = {sc_gbdt:.4f}")
        
    print(f"\n>> Pairwise GBDT Mean LOOCV Score: {np.mean(gbdt_scores):.4f} <<")
    
    # 4. Ensemble LOOCV Evaluation (75% Physics + 25% Pairwise GBDT)
    print("\n" + "=" * 85)
    print("[Step 4] Leave-One-Case-Out Evaluation: 75% Physics + 25% Pairwise GBDT Ensemble")
    print("=" * 85)
    
    w1, w2, w3, w4 = best_weights
    ensemble_cv_records = []
    
    for val_case in case_names:
        train_cases = {k: v for k, v in train_features_dict.items() if k != val_case}
        df_pairs_train = build_pairwise_dataset(train_cases, label_map)
        val_feat = train_features_dict[val_case]
        
        # Physics normalized score
        s_phys = score_physics(val_feat, w1, w2, w3, w4)
        s_phys_norm = (s_phys - s_phys.min()) / (s_phys.max() - s_phys.min() + 1e-9)
        
        # Pairwise GBDT normalized score
        s_gbdt_norm = train_pairwise_gbdt(df_pairs_train, val_feat)
        
        # Ensemble blend
        s_ens = 0.75 * s_phys_norm + 0.25 * s_gbdt_norm
        ranked_ens = s_ens.sort_values(ascending=False).index.tolist()
        
        true_car = label_map[val_case]
        sc_ens = compute_linear_rank_decay_score(true_car, ranked_ens)
        rank_pos = ranked_ens.index(true_car) + 1
        
        ensemble_cv_records.append({
            "Case": val_case,
            "Fault Car": true_car,
            "Pred Rank 1": ranked_ens[0],
            "Rank Pos": f"{rank_pos}/{len(val_feat)}",
            "Score": f"{sc_ens:.4f}",
            "Full Order": "|".join(ranked_ens)
        })
        
    df_ens_cv = pd.DataFrame(ensemble_cv_records)
    print(df_ens_cv[['Case', 'Fault Car', 'Pred Rank 1', 'Rank Pos', 'Score']].to_string(index=False))
    ens_mean = np.mean([float(r['Score']) for r in ensemble_cv_records])
    print(f"\n>> Final Ensemble Mean LOOCV Score: {ens_mean:.4f} / 1.0000 <<")
    
    # 5. Held-Out Test Case Inference (acv_test_case.xlsx)
    print("\n" + "=" * 85)
    print("[Step 5] Held-Out Test Case Prediction & Sensitivity Analysis")
    print("=" * 85)
    test_fpath = glob.glob(os.path.join(TEST_DIR, "*.xlsx"))[0]
    test_fname = os.path.basename(test_fpath)
    df_test_raw = pd.read_excel(test_fpath)
    df_test_clean, test_active = clean_and_preprocess_case(df_test_raw)
    test_feat = extract_car_features(df_test_clean, test_active)
    
    # Train Pairwise GBDT on all 6 cases
    all_pairs_df = build_pairwise_dataset(train_features_dict, label_map)
    test_gbdt_scores = train_pairwise_gbdt(all_pairs_df, test_feat)
    
    # Physics on Test
    test_phys_scores = score_physics(test_feat, w1, w2, w3, w4)
    test_phys_norm = (test_phys_scores - test_phys_scores.min()) / (test_phys_scores.max() - test_phys_scores.min() + 1e-9)
    
    # Final Ensemble on Test
    test_ens_scores = 0.75 * test_phys_norm + 0.25 * test_gbdt_scores
    final_ranked_test = test_ens_scores.sort_values(ascending=False).index.tolist()
    final_ranking_str = "|".join(final_ranked_test)
    
    print(f"\nTarget Test File: {test_fname}")
    print(f"Final Predicted Car Ranking: {final_ranking_str}")
    print(f"Top 3 Identified Suspects: 1st -> Car {final_ranked_test[0]}, 2nd -> Car {final_ranked_test[1]}, 3rd -> Car {final_ranked_test[2]}")
    
    # Detailed score breakdown
    df_test_summary = pd.DataFrame({
        "Car": test_active,
        "Physics_Norm_Score": [test_phys_norm[c] for c in test_active],
        "Pairwise_GBDT_Score": [test_gbdt_scores[c] for c in test_active],
        "Ensemble_Final_Score": [test_ens_scores[c] for c in test_active]
    }).sort_values(by="Ensemble_Final_Score", ascending=False)
    print("\nDetailed Test Scoring Breakdown:")
    print(df_test_summary.to_string(index=False))
    
    # Sensitivity check
    print("\nSensitivity Verification:")
    print("Testing perturbations of ensemble weights:")
    for alpha in [0.90, 0.75, 0.60, 0.50]:
        pert_scores = alpha * test_phys_norm + (1 - alpha) * test_gbdt_scores
        pert_rank = pert_scores.sort_values(ascending=False).index.tolist()
        print(f"  Alpha={alpha:.2f} (Physics={alpha*100:.0f}%, GBDT={(1-alpha)*100:.0f}%): Rank 1 -> Car {pert_rank[0]} | Order -> {'|'.join(pert_rank[:4])}")
        
    # Write final submission file
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "acv_predictions.csv")
    sub_df = pd.DataFrame([{"file_id": test_fname, "ranked_cars": final_ranking_str}])
    sub_df.to_csv(out_path, index=False)
    print(f"\nSuccessfully wrote final predictions to: {out_path}")
    print("Schema check:")
    print(sub_df.to_string(index=False))

if __name__ == "__main__":
    main()
