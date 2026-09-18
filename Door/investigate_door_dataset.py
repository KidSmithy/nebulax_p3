"""
Door Subsystem Comprehensive Investigation, Profiling, and Modeling Suite
==========================================================================
This script supplements the Door documentation files:
- PS3/02_Datasets/Door/DOOR_DATASET_ANALYSIS.md
- PS3/03_References/Door/DOOR_REFERENCES_AND_SPECS.md
- PS3/DOOR_MASTER_MODELING_GUIDE.md

Capabilities:
1. Performs full dataset integrity and schema inspection.
2. Mathematically segments continuous telemetry streams (Train and Test) using timestamp jump detection (dt > 0.1s).
3. Conducts rigorous statistical hypothesis tests (Mann-Whitney U, Cohen's d) between Normal and Abnormal resistance.
4. Extracts 30 domain-specific physical and kinematic features (motor current, voltage, EMF, power, energy, switches).
5. Runs 5-Fold Stratified Cross-Validation on 5 model architectures using the official IoU-weighted F1 metric.
6. Generates high-confidence predictions on Test.csv formatted exactly to competition specifications (door_predictions.csv).
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
candidate_data_dirs = [
    os.path.join(BASE_DIR, "02_Datasets", "Door"),
    os.path.join(BASE_DIR, "..", "PS3", "02_Datasets", "Door"),
    os.path.join(BASE_DIR, "PS3", "02_Datasets", "Door"),
    r"c:\Users\Rald999\Documents\GitHub\nebulax_p3\PS3\02_Datasets\Door",
]
DOOR_DATA_DIR = next((d for d in candidate_data_dirs if os.path.exists(d)), candidate_data_dirs[0])
DOOR_REF_DIR = os.path.join(BASE_DIR, "03_References", "Door")

def parse_series_datetimes(series):
    """
    Fast vectorized parsing of timestamp format: YYYY-M-D-H-M-S-ms
    e.g., '2023-7-5-0-0-3-760'
    """
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
    """
    Intersection-over-Union (IoU) of two 1D temporal intervals.
    """
    inter = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    union = (end_a - start_a) + (end_b - start_b) - inter
    return inter / union if union > 0 else 0.0

def evaluate_iou_weighted_f1(true_df, pred_df):
    """
    Official IoU-weighted F1 scoring function (Door Info Kit Section 4).
    1. Segments must match on predicted label.
    2. Greedy bipartite matching in descending order of IoU.
    3. Harmonic mean of soft recall and soft precision.
    """
    candidates = []
    for p_idx, p_row in pred_df.iterrows():
        p_s = p_row['start_sec']
        p_e = p_row['end_sec']
        p_lbl = p_row['prediction']
        
        for t_idx, t_row in true_df.iterrows():
            t_s = t_row['start_sec']
            t_e = t_row['end_sec']
            t_lbl = t_row['status']
            
            if p_lbl == t_lbl:
                iou = compute_iou(p_s, p_e, t_s, t_e)
                if iou > 0:
                    candidates.append((iou, p_idx, t_idx))
                    
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    matched_preds = set()
    matched_trues = set()
    sum_iou = 0.0
    
    for iou, p_idx, t_idx in candidates:
        if p_idx not in matched_preds and t_idx not in matched_trues:
            matched_preds.add(p_idx)
            matched_trues.add(t_idx)
            sum_iou += iou
            
    n_true = len(true_df)
    n_pred = len(pred_df)
    
    soft_recall = sum_iou / n_true if n_true > 0 else 0.0
    soft_precision = sum_iou / n_pred if n_pred > 0 else 0.0
    
    if soft_recall + soft_precision > 0:
        score = 2 * soft_recall * soft_precision / (soft_recall + soft_precision)
    else:
        score = 0.0
        
    return score, soft_recall, soft_precision

def detect_segments(df_stream):
    """
    Detects individual door cycles by finding timestamp jumps (dt > 0.1s).
    """
    dt_series = df_stream['parsed_dt']
    gaps = np.where(dt_series.diff().dt.total_seconds() > 0.1)[0]
    starts = [0] + list(gaps)
    ends = [g - 1 for g in gaps] + [len(df_stream) - 1]
    return starts, ends

def extract_features_from_segments(df_stream, starts, ends, df_ground_truth=None):
    """
    Extracts 30 physical, electrical, and statistical features for each detected segment.
    """
    dt_series = df_stream['parsed_dt']
    records = []
    
    for i in range(len(starts)):
        s, e = starts[i], ends[i]
        sub = df_stream.iloc[s:e+1]
        
        cur = sub['Motor current(mA)'].values
        volt = sub['Motor Voltage(10mV)'].values
        emf = sub['Motor electrodynamic force'].values
        pos = sub['Door leaf position'].values
        
        pos_diff = pos[-1] - pos[0]
        inferred_op = 'Open' if pos_diff > 0 else 'Close'
        speed = np.abs(np.diff(pos) / 0.02)
        
        # Power & Energy (V in 10mV -> divide by 100 to get Volts, cur in mA -> divide by 1000 for Amps)
        # Power in Watts = (volt / 100.0) * (cur / 1000.0) = volt * cur / 100000.0
        power = (volt / 100.0) * (cur / 1000.0)
        mech_power = (emf / 100.0) * (cur / 1000.0)
        
        s_dt = dt_series.iloc[s]
        e_dt = dt_series.iloc[e]
        duration_s = (e_dt - s_dt).total_seconds()
        
        rec = {
            'seg_idx': i,
            'start_time': df_stream['Datetime'].iloc[s],
            'end_time': df_stream['Datetime'].iloc[e],
            'start_sec': s_dt.timestamp(),
            'end_sec': e_dt.timestamp(),
            'duration_s': duration_s,
            'n_rows': len(sub),
            'op_is_open': 1 if inferred_op == 'Open' else 0,
            
            # Current Features
            'cur_mean': np.mean(cur),
            'cur_std': np.std(cur),
            'cur_min': np.min(cur),
            'cur_max': np.max(cur),
            'cur_median': np.median(cur),
            'cur_q75': np.percentile(cur, 75),
            'cur_q90': np.percentile(cur, 90),
            'cur_rms': np.sqrt(np.mean(cur**2)),
            'cur_integral': np.sum(np.abs(cur) * 0.02),
            
            # Voltage Features
            'volt_mean': np.mean(volt),
            'volt_std': np.std(volt),
            'volt_min': np.min(volt),
            'volt_max': np.max(volt),
            'volt_q75': np.percentile(volt, 75),
            
            # EMF Features
            'emf_mean': np.mean(emf),
            'emf_std': np.std(emf),
            'emf_min': np.min(emf),
            'emf_max': np.max(emf),
            'emf_q75': np.percentile(emf, 75),
            
            # Energy & Power
            'work_elec': np.sum(np.abs(power) * 0.02),
            'work_mech': np.sum(np.abs(mech_power) * 0.02),
            'power_max': np.max(np.abs(power)),
            
            # Kinematics
            'speed_mean': np.mean(speed) if len(speed) > 0 else 0,
            'speed_max': np.max(speed) if len(speed) > 0 else 0,
            'speed_std': np.std(speed) if len(speed) > 0 else 0,
            
            # Switch Transitions
            'switch_dc_trans': np.sum(np.abs(np.diff(sub['DCSR'].values))),
            'switch_dl_trans': np.sum(np.abs(np.diff(sub['DLSR'].values))),
        }
        
        if df_ground_truth is not None:
            rec['status'] = df_ground_truth['status'].iloc[i]
            rec['operation'] = df_ground_truth['operation'].iloc[i]
            rec['target'] = 1 if df_ground_truth['status'].iloc[i] == 'Abnormal resistance' else 0
            
        records.append(rec)
        
    return pd.DataFrame(records)

def main():
    print("=" * 85)
    print("    DOOR SUBSYSTEM: DATA INVESTIGATION, STATISTICAL AUDIT & MODEL BENCHMARK")
    print("=" * 85)
    
    # 1. Load Files
    train_path = os.path.join(DOOR_DATA_DIR, "Train.csv")
    ans_path = os.path.join(DOOR_DATA_DIR, "Train_Segments_Answer.csv")
    test_path = os.path.join(DOOR_DATA_DIR, "Test.csv")
    
    df_train = pd.read_csv(train_path)
    df_ans = pd.read_csv(ans_path)
    df_test = pd.read_csv(test_path)
    
    df_train['parsed_dt'] = parse_series_datetimes(df_train['Datetime'])
    df_test['parsed_dt'] = parse_series_datetimes(df_test['Datetime'])
    
    print(f"\n[1] DATASET OVERVIEW:")
    print(f"  • Train.csv: {len(df_train):,} rows, {df_train.shape[1]} columns, span: {df_train['parsed_dt'].min()} to {df_train['parsed_dt'].max()}")
    print(f"  • Train_Segments_Answer.csv: {len(df_ans)} segments labeled")
    print(f"  • Test.csv: {len(df_test):,} rows, {df_test.shape[1]} columns, span: {df_test['parsed_dt'].min()} to {df_test['parsed_dt'].max()}")
    
    # 2. Segment Detection Proof
    train_starts, train_ends = detect_segments(df_train)
    test_starts, test_ends = detect_segments(df_test)
    
    # Verify exact match
    perfect_match = True
    for i in range(len(df_ans)):
        s_time = df_train['Datetime'].iloc[train_starts[i]]
        e_time = df_train['Datetime'].iloc[train_ends[i]]
        if s_time != df_ans['start_time'].iloc[i] or e_time != df_ans['end_time'].iloc[i]:
            perfect_match = False
            break
            
    print(f"\n[2] CYCLE BOUNDARY DETECTION:")
    print(f"  • Train contiguous segments found (dt > 0.1s): {len(train_starts)}")
    print(f"  • Ground-truth segments in answer file: {len(df_ans)}")
    print(f"  • 100.0% Exact Timestamp & Row Boundary Match: {perfect_match}")
    print(f"  • Test contiguous segments found (dt > 0.1s): {len(test_starts)} candidate cycles")
    
    # 3. Feature Extraction
    train_feat = extract_features_from_segments(df_train, train_starts, train_ends, df_ans)
    test_feat = extract_features_from_segments(df_test, test_starts, test_ends)
    
    feature_cols = [c for c in train_feat.columns if c not in [
        'seg_idx', 'start_time', 'end_time', 'start_sec', 'end_sec', 
        'status', 'operation', 'target'
    ]]
    
    # 4. Statistical Hypothesis Testing
    print(f"\n[3] STATISTICAL DISCRIMINATION ANALYSIS (Mann-Whitney U & Cohen's d):")
    for op in ['Close', 'Open']:
        sub_df = train_feat[train_feat['operation'] == op]
        norm = sub_df[sub_df['target'] == 0]
        abn = sub_df[sub_df['target'] == 1]
        
        results = []
        for f in ['cur_integral', 'cur_mean', 'cur_q75', 'cur_min', 'volt_max', 'volt_mean', 'emf_std', 'work_elec']:
            m_norm, s_norm = norm[f].mean(), norm[f].std()
            m_abn, s_abn = abn[f].mean(), abn[f].std()
            u_stat, p_val = stats.mannwhitneyu(abn[f], norm[f], alternative='two-sided')
            pooled_std = np.sqrt(((len(norm)-1)*s_norm**2 + (len(abn)-1)*s_abn**2) / (len(norm) + len(abn) - 2))
            cohen_d = (m_abn - m_norm) / pooled_std if pooled_std > 0 else 0
            results.append({'Feature': f, 'Normal Mean': round(m_norm, 2), 'Abnormal Mean': round(m_abn, 2), 'p-value': f"{p_val:.2e}", "Cohen's d": round(cohen_d, 2)})
            
        print(f"\n  --- Operation: {op.upper()} ({len(norm)} Normal, {len(abn)} Abnormal) ---")
        print(pd.DataFrame(results).to_string(index=False))

    # 5. Model Benchmarking via 5-Fold Stratified CV
    print(f"\n[4] 5-FOLD STRATIFIED CV BENCHMARK (Official IoU-Weighted F1 Metric):")
    X = train_feat[feature_cols].values
    y = train_feat['target'].values
    strat_key = train_feat['operation'] + "_" + train_feat['target'].astype(str)
    
    models = {
        'Random Forest (100 trees)': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        'XGBoost Classifier': xgb.XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.08, eval_metric='logloss', random_state=42),
        'Logistic Regression (L2)': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        'Support Vector Machine (RBF)': SVC(C=10.0, probability=True, random_state=42),
        'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=100, max_depth=4, random_state=42)
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    benchmark_rows = []
    
    for name, clf in models.items():
        oof_preds = np.zeros(len(train_feat))
        oof_probs = np.zeros(len(train_feat))
        
        for train_idx, val_idx in skf.split(X, strat_key):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_va = X[val_idx], y[val_idx]
            
            if 'Regression' in name or 'SVM' in name:
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_va = scaler.transform(X_va)
                
            clf.fit(X_tr, y_tr)
            oof_preds[val_idx] = clf.predict(X_va)
            if hasattr(clf, "predict_proba"):
                oof_probs[val_idx] = clf.predict_proba(X_va)[:, 1]
                
        pred_sub_df = train_feat[['start_sec', 'end_sec']].copy()
        pred_sub_df['prediction'] = ['Abnormal resistance' if p == 1 else 'Normal' for p in oof_preds]
        true_sub_df = train_feat[['start_sec', 'end_sec', 'status']].copy()
        
        iou_f1, soft_rec, soft_prec = evaluate_iou_weighted_f1(true_sub_df, pred_sub_df)
        acc = accuracy_score(y, oof_preds)
        prec = precision_score(y, oof_preds, zero_division=0)
        rec = recall_score(y, oof_preds, zero_division=0)
        auc = roc_auc_score(y, oof_probs) if len(np.unique(oof_probs)) > 1 else 0.5
        
        benchmark_rows.append({
            'Model': name,
            'IoU-Weighted F1': round(iou_f1, 4),
            'Soft Recall': round(soft_rec, 4),
            'Soft Precision': round(soft_prec, 4),
            'Accuracy': round(acc, 4),
            'ROC-AUC': round(auc, 4)
        })
        
    df_bm = pd.DataFrame(benchmark_rows).sort_values(by='IoU-Weighted F1', ascending=False)
    print(df_bm.to_string(index=False))

    # 6. Test Inference & Submission File Generation
    print(f"\n[5] TEST INFERENCE & PREDICTION GENERATION:")
    top_model = models['Random Forest (100 trees)']
    top_model.fit(X, y)
    
    X_test = test_feat[feature_cols].values
    test_preds = top_model.predict(X_test)
    test_probs = top_model.predict_proba(X_test)[:, 1]
    
    test_feat['prediction'] = ['Abnormal resistance' if p == 1 else 'Normal' for p in test_preds]
    test_feat['confidence'] = test_probs
    
    sub_df = test_feat[['start_time', 'end_time', 'prediction']].copy()
    sub_path = os.path.join(BASE_DIR, "door_predictions.csv")
    sub_df.to_csv(sub_path, index=False)
    
    print(f"  • Saved predictions to: {sub_path}")
    print(f"  • Total test segments: {len(sub_df)}")
    print(f"  • Predicted Class Counts:\n{sub_df['prediction'].value_counts().to_string()}")
    print("\nFirst 5 Predicted Rows:")
    print(sub_df.head().to_string(index=False))
    print("=" * 85)

if __name__ == "__main__":
    main()
