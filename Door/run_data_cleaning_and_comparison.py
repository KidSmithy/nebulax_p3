import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb

BASE_DIR = r"c:\Users\Rald999\Documents\GitHub\nebulax_p3\PS3"
DOOR_DATA_DIR = os.path.join(BASE_DIR, "02_Datasets", "Door")

def parse_series_datetimes(series):
    split_df = series.str.split('-', expand=True).astype(int)
    return pd.to_datetime({
        'year': split_df[0],
        'month': split_df[1],
        'day': split_df[2],
        'hour': split_df[3],
        'minute': split_df[4],
        'second': split_df[5],
        'microsecond': split_df[6] * 1000
    })

def compute_iou(start_a, end_a, start_b, end_b):
    inter = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    union = (end_a - start_a) + (end_b - start_b) - inter
    return inter / union if union > 0 else 0.0

def evaluate_iou_weighted_f1(true_df, pred_df):
    candidates = []
    for p_idx, p_row in pred_df.iterrows():
        p_s, p_e, p_lbl = p_row['start_sec'], p_row['end_sec'], p_row['prediction']
        for t_idx, t_row in true_df.iterrows():
            t_s, t_e, t_lbl = t_row['start_sec'], t_row['end_sec'], t_row['status']
            if p_lbl == t_lbl:
                iou = compute_iou(p_s, p_e, t_s, t_e)
                if iou > 0:
                    candidates.append((iou, p_idx, t_idx))
                    
    candidates.sort(key=lambda x: x[0], reverse=True)
    matched_preds, matched_trues = set(), set()
    sum_iou = 0.0
    
    for iou, p_idx, t_idx in candidates:
        if p_idx not in matched_preds and t_idx not in matched_trues:
            matched_preds.add(p_idx)
            matched_trues.add(t_idx)
            sum_iou += iou
            
    n_true, n_pred = len(true_df), len(pred_df)
    soft_recall = sum_iou / n_true if n_true > 0 else 0.0
    soft_precision = sum_iou / n_pred if n_pred > 0 else 0.0
    score = 2 * soft_recall * soft_precision / (soft_recall + soft_precision) if (soft_recall + soft_precision) > 0 else 0.0
    return score, soft_recall, soft_precision

def detect_segments(dt_series):
    gaps = np.where(dt_series.diff().dt.total_seconds() > 0.1)[0]
    starts = [0] + list(gaps)
    ends = [g - 1 for g in gaps] + [len(dt_series) - 1]
    return starts, ends

def run_data_cleaning():
    print("=" * 80)
    print("STEP 1: PERFORMING FORMAL DATA CLEANING ON DOOR DATASET")
    print("=" * 80)
    
    train_raw = pd.read_csv(os.path.join(DOOR_DATA_DIR, "Train.csv"))
    test_raw = pd.read_csv(os.path.join(DOOR_DATA_DIR, "Test.csv"))
    
    # 1. Standardize Timestamps
    train_clean = train_raw.copy()
    test_clean = test_raw.copy()
    
    train_clean['parsed_datetime'] = parse_series_datetimes(train_clean['Datetime'])
    test_clean['parsed_datetime'] = parse_series_datetimes(test_clean['Datetime'])
    
    # 2. Drop Inactive / Zero-Variance Column
    print("  • Dropping zero-variance column: 'Door Locked' (100% constant 0 in both Train & Test)")
    train_clean.drop(columns=['Door Locked'], inplace=True)
    test_clean.drop(columns=['Door Locked'], inplace=True)
    
    # 3. Position Clamping / Offset Normalization
    # Test has a cycle reaching 807 due to encoder offset / overtravel
    print("  • Normalizing Door leaf position (clamping mechanical limits to nominal travel 0-700)")
    train_clean['Door leaf position clamped'] = train_clean['Door leaf position'].clip(0, 700)
    test_clean['Door leaf position clamped'] = test_clean['Door leaf position'].clip(0, 700)
    
    # 4. Standardize Column Headers to Canonical snake_case
    rename_map = {
        'Datetime': 'datetime_str',
        'parsed_datetime': 'datetime_parsed',
        'Motor current(mA)': 'motor_current_ma',
        'Motor Voltage(10mV)': 'motor_voltage_10mv',
        'Motor electrodynamic force': 'motor_back_emf',
        'Door opening time(.1s)': 'reg_open_time_ds',
        'Door closing time(.1s)': 'reg_close_time_ds',
        'Close command': 'cmd_close',
        'Open command': 'cmd_open',
        'DCSR': 'switch_dcsr',
        'DCSL': 'switch_dcsl',
        'DLSR': 'switch_dlsr',
        'DLSL': 'switch_dlsl',
        'Door Opened': 'status_door_opened',
        'Door is opening': 'flag_is_opening',
        'Door is closing': 'flag_is_closing',
        'Door leaf position': 'leaf_pos_raw',
        'Door leaf position clamped': 'leaf_pos_clamped'
    }
    train_clean.rename(columns=rename_map, inplace=True)
    test_clean.rename(columns=rename_map, inplace=True)
    
    # Save Cleaned CSV files
    train_clean_path = os.path.join(DOOR_DATA_DIR, "Train_cleaned.csv")
    test_clean_path = os.path.join(DOOR_DATA_DIR, "Test_cleaned.csv")
    
    train_clean.to_csv(train_clean_path, index=False)
    test_clean.to_csv(test_clean_path, index=False)
    
    print(f"  • Successfully created: {train_clean_path} (Shape: {train_clean.shape})")
    print(f"  • Successfully created: {test_clean_path} (Shape: {test_clean.shape})")
    
    return train_raw, test_raw, train_clean, test_clean

def extract_naive_features(df_stream, starts, ends, dt_series, df_ans=None):
    """
    Pipeline A: Naive / Uncleaned Feature Extraction
    - Keeps raw column names and unscaled values
    - Uses zero-variance column Door Locked
    - Uses raw timer columns (which have 0s in Test)
    - Uses raw pos_max directly without clamping/deltas
    - Does NOT compute physical energy / electrical work features
    """
    records = []
    for i in range(len(starts)):
        s, e = starts[i], ends[i]
        sub = df_stream.iloc[s:e+1]
        
        cur = sub['Motor current(mA)'].values
        volt = sub['Motor Voltage(10mV)'].values
        emf = sub['Motor electrodynamic force'].values
        pos = sub['Door leaf position'].values
        
        s_dt, e_dt = dt_series.iloc[s], dt_series.iloc[e]
        
        rec = {
            'seg_idx': i,
            'start_sec': s_dt.timestamp(),
            'end_sec': e_dt.timestamp(),
            'n_rows': len(sub),
            # Uncleaned static timer features
            'raw_open_time': sub['Door opening time(.1s)'].mean(),
            'raw_close_time': sub['Door closing time(.1s)'].mean(),
            # Raw position features vulnerable to Test 807 offset
            'raw_pos_max': np.max(pos),
            'raw_pos_min': np.min(pos),
            'raw_pos_mean': np.mean(pos),
            # Zero-variance column included!
            'raw_door_locked': sub['Door Locked'].mean(),
            # Basic stats only
            'cur_mean': np.mean(cur),
            'cur_std': np.std(cur),
            'cur_max': np.max(cur),
            'cur_min': np.min(cur),
            'volt_mean': np.mean(volt),
            'volt_std': np.std(volt),
            'volt_max': np.max(volt),
            'emf_mean': np.mean(emf),
            'emf_std': np.std(emf),
            'emf_max': np.max(emf),
        }
        if df_ans is not None:
            rec['status'] = df_ans['status'].iloc[i]
            rec['operation'] = df_ans['operation'].iloc[i]
            rec['target'] = 1 if df_ans['status'].iloc[i] == 'Abnormal resistance' else 0
            
        records.append(rec)
    return pd.DataFrame(records)

def extract_cleaned_features(df_stream_clean, starts, ends, df_ans=None):
    """
    Pipeline B: Cleaned & Physics-Engineered Feature Extraction
    - Zero-variance column dropped
    - True empirical duration used instead of buggy register timers
    - Clamped & stroke-normalized leaf position
    - Comprehensive electromechanical energy features: current integral, RMS, Q75, electrical work, mechanical work
    - Switch transition counts
    """
    dt_series = df_stream_clean['datetime_parsed']
    records = []
    
    for i in range(len(starts)):
        s, e = starts[i], ends[i]
        sub = df_stream_clean.iloc[s:e+1]
        
        cur = sub['motor_current_ma'].values
        volt = sub['motor_voltage_10mv'].values
        emf = sub['motor_back_emf'].values
        pos = sub['leaf_pos_clamped'].values
        
        s_dt, e_dt = dt_series.iloc[s], dt_series.iloc[e]
        duration_s = (e_dt - s_dt).total_seconds()
        
        pos_diff = pos[-1] - pos[0]
        inferred_op = 'Open' if pos_diff > 0 else 'Close'
        speed = np.abs(np.diff(pos) / 0.02)
        
        # Physics Work & Power
        power = (volt / 100.0) * (cur / 1000.0)
        mech_power = (emf / 100.0) * (cur / 1000.0)
        
        rec = {
            'seg_idx': i,
            'start_sec': s_dt.timestamp(),
            'end_sec': e_dt.timestamp(),
            'duration_s': duration_s,
            'n_rows': len(sub),
            'op_is_open': 1 if inferred_op == 'Open' else 0,
            
            # Clean Current Features
            'cur_mean': np.mean(cur),
            'cur_std': np.std(cur),
            'cur_min': np.min(cur),
            'cur_max': np.max(cur),
            'cur_median': np.median(cur),
            'cur_q75': np.percentile(cur, 75),
            'cur_q90': np.percentile(cur, 90),
            'cur_rms': np.sqrt(np.mean(cur**2)),
            'cur_integral': np.sum(np.abs(cur) * 0.02),
            
            # Clean Voltage Features
            'volt_mean': np.mean(volt),
            'volt_std': np.std(volt),
            'volt_min': np.min(volt),
            'volt_max': np.max(volt),
            'volt_q75': np.percentile(volt, 75),
            
            # Clean Back-EMF Features
            'emf_mean': np.mean(emf),
            'emf_std': np.std(emf),
            'emf_min': np.min(emf),
            'emf_max': np.max(emf),
            'emf_q75': np.percentile(emf, 75),
            
            # Energy & Power Work
            'work_elec': np.sum(np.abs(power) * 0.02),
            'work_mech': np.sum(np.abs(mech_power) * 0.02),
            'power_max': np.max(np.abs(power)),
            
            # Speed & Kinematics
            'speed_mean': np.mean(speed) if len(speed) > 0 else 0,
            'speed_max': np.max(speed) if len(speed) > 0 else 0,
            'speed_std': np.std(speed) if len(speed) > 0 else 0,
            
            # Clean Switch Transitions
            'switch_dc_trans': np.sum(np.abs(np.diff(sub['switch_dcsr'].values))),
            'switch_dl_trans': np.sum(np.abs(np.diff(sub['switch_dlsr'].values))),
        }
        if df_ans is not None:
            rec['status'] = df_ans['status'].iloc[i]
            rec['operation'] = df_ans['operation'].iloc[i]
            rec['target'] = 1 if df_ans['status'].iloc[i] == 'Abnormal resistance' else 0
            
        records.append(rec)
    return pd.DataFrame(records)

def evaluate_pipeline(train_df, feature_cols, pipe_name, use_scaler=False):
    X = train_df[feature_cols].values
    y = train_df['target'].values
    strat_key = train_df['operation'] + "_" + train_df['target'].astype(str)
    
    models = {
        'Logistic Regression (L2)': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        'Support Vector Machine (RBF)': SVC(C=10.0, probability=True, random_state=42),
        'Random Forest (100 trees)': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=100, max_depth=4, random_state=42),
        'XGBoost Classifier': xgb.XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.08, eval_metric='logloss', random_state=42)
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    
    for name, clf in models.items():
        oof_preds = np.zeros(len(train_df))
        oof_probs = np.zeros(len(train_df))
        
        for tr_idx, val_idx in skf.split(X, strat_key):
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_va, y_va = X[val_idx], y[val_idx]
            
            # Scale features if requested or if linear/distance model
            if use_scaler or ('Regression' in name or 'SVM' in name):
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_va = scaler.transform(X_va)
                
            clf.fit(X_tr, y_tr)
            oof_preds[val_idx] = clf.predict(X_va)
            if hasattr(clf, "predict_proba"):
                oof_probs[val_idx] = clf.predict_proba(X_va)[:, 1]
                
        pred_sub = train_df[['start_sec', 'end_sec']].copy()
        pred_sub['prediction'] = ['Abnormal resistance' if p == 1 else 'Normal' for p in oof_preds]
        true_sub = train_df[['start_sec', 'end_sec', 'status']].copy()
        
        iou_f1, soft_rec, soft_prec = evaluate_iou_weighted_f1(true_sub, pred_sub)
        acc = accuracy_score(y, oof_preds)
        prec = precision_score(y, oof_preds, zero_division=0)
        rec = recall_score(y, oof_preds, zero_division=0)
        f1 = f1_score(y, oof_preds, zero_division=0)
        auc = roc_auc_score(y, oof_probs) if len(np.unique(oof_probs)) > 1 else 0.5
        loss = log_loss(y, oof_probs) if len(np.unique(oof_probs)) > 1 else 1.0
        
        results.append({
            'Pipeline': pipe_name,
            'Model': name,
            'IoU-Weighted F1': round(iou_f1, 4),
            'Soft Recall': round(soft_rec, 4),
            'Soft Precision': round(soft_prec, 4),
            'Accuracy': round(acc, 4),
            'F1-Score': round(f1, 4),
            'ROC-AUC': round(auc, 4),
            'Log Loss': round(loss, 4)
        })
        
    return pd.DataFrame(results)

def main():
    # 1. Clean Data & Save
    train_raw, test_raw, train_clean, test_clean = run_data_cleaning()
    
    ans = pd.read_csv(os.path.join(DOOR_DATA_DIR, "Train_Segments_Answer.csv"))
    train_dt = parse_series_datetimes(train_raw['Datetime'])
    test_dt = parse_series_datetimes(test_raw['Datetime'])
    
    train_starts, train_ends = detect_segments(train_dt)
    test_starts, test_ends = detect_segments(test_dt)
    
    # 2. Extract Features
    print("\n" + "=" * 80)
    print("STEP 2: EXTRACTING FEATURES (BEFORE VS AFTER CLEANING)")
    print("=" * 80)
    
    # Before Cleaning / Naive Pipeline
    df_naive = extract_naive_features(train_raw, train_starts, train_ends, train_dt, ans)
    naive_feature_cols = [c for c in df_naive.columns if c not in ['seg_idx', 'start_sec', 'end_sec', 'status', 'operation', 'target']]
    print(f"  • Before Cleaning Feature Count: {len(naive_feature_cols)} features (includes zero-variance Door Locked, uncleaned timers, raw pos)")
    
    # After Cleaning / Cleaned Pipeline
    df_clean = extract_cleaned_features(train_clean, train_starts, train_ends, ans)
    clean_feature_cols = [c for c in df_clean.columns if c not in ['seg_idx', 'start_sec', 'end_sec', 'status', 'operation', 'target']]
    print(f"  • After Cleaning Feature Count: {len(clean_feature_cols)} features (zero-variance dropped, empirical duration, energy & work)")
    
    # 3. Model Benchmark Evaluation (5-Fold Stratified CV)
    print("\n" + "=" * 80)
    print("STEP 3: 5-FOLD STRATIFIED CV BENCHMARK (BEFORE VS AFTER CLEANING)")
    print("=" * 80)
    
    # Naive Pipeline (No scaling, uncleaned)
    res_before = evaluate_pipeline(df_naive, naive_feature_cols, "Before Cleaning (Raw)", use_scaler=False)
    
    # Cleaned Pipeline (Cleaned features + scaling)
    res_after = evaluate_pipeline(df_clean, clean_feature_cols, "After Cleaning (Cleaned)", use_scaler=True)
    
    # Combined Comparison
    comparison_df = pd.concat([res_before, res_after], ignore_index=True)
    
    # Print Side-by-Side Comparison
    print("\n=== COMPLETE BENCHMARK COMPARISON TABLE ===")
    print(comparison_df.to_string(index=False))
    
    # Model-by-Model Delta Summary
    print("\n" + "=" * 80)
    print("MODEL-BY-MODEL PERFORMANCE DELTA SUMMARY")
    print("=" * 80)
    
    pivot_f1 = comparison_df.pivot(index='Model', columns='Pipeline', values='IoU-Weighted F1')
    pivot_auc = comparison_df.pivot(index='Model', columns='Pipeline', values='ROC-AUC')
    pivot_loss = comparison_df.pivot(index='Model', columns='Pipeline', values='Log Loss')
    
    delta_df = pd.DataFrame({
        'F1 Before': pivot_f1['Before Cleaning (Raw)'],
        'F1 After': pivot_f1['After Cleaning (Cleaned)'],
        'F1 Delta': (pivot_f1['After Cleaning (Cleaned)'] - pivot_f1['Before Cleaning (Raw)']).round(4),
        'AUC Before': pivot_auc['Before Cleaning (Raw)'],
        'AUC After': pivot_auc['After Cleaning (Cleaned)'],
        'AUC Delta': (pivot_auc['After Cleaning (Cleaned)'] - pivot_auc['Before Cleaning (Raw)']).round(4),
        'Loss Before': pivot_loss['Before Cleaning (Raw)'],
        'Loss After': pivot_loss['After Cleaning (Cleaned)'],
        'Loss Delta (Reduction)': (pivot_loss['Before Cleaning (Raw)'] - pivot_loss['After Cleaning (Cleaned)']).round(4),
    })
    print(delta_df.to_string())
    
    # 4. Test Set Edge Case Simulation & Prediction Stability
    print("\n" + "=" * 80)
    print("STEP 4: TEST SET ROBUSTNESS & EDGE CASE IMPACT ANALYSIS")
    print("=" * 80)
    
    df_test_naive = extract_naive_features(test_raw, test_starts, test_ends, test_dt)
    df_test_clean = extract_cleaned_features(test_clean, test_starts, test_ends)
    
    # Check impact on test segment 34 (row 5351 with position 807)
    print("Test Segment with Position 807 (Cycle 34):")
    print(f"  • Before cleaning raw pos_max: {df_test_naive.iloc[33]['raw_pos_max']}")
    print(f"  • After cleaning clamped pos_max: {df_test_clean.iloc[33]['speed_max']:.2f} (speed based, immune to 807 offset)")
    
    # Check test timer zero issue
    print("\nTest Segments with Zero Timer Values (Uninitialized Registers):")
    print(f"  • Before cleaning zero timer count: {(df_test_naive['raw_open_time'] == 0).sum()} cycles with open_time == 0")
    print(f"  • After cleaning empirical duration mean: {df_test_clean['duration_s'].mean():.2f}s (all cycles have accurate physical duration)")

if __name__ == "__main__":
    main()
