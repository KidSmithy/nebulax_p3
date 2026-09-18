"""
SHM (Structural Health Monitoring) Dataset Investigation & Feature Profiling Script
===================================================================================
Subsystem: Rail Vehicle Structural Health Monitoring - Cumulative Fatigue Damage Regression
Target: c:/Users/Rald999/Documents/GitHub/nebulax_p3/SHM/

Features:
1. Universal path detection for datasets and labels
2. O(N) stack-based ASTM E1049-85 rainflow cycle counter
3. Multi-order Basquin stress moments (E[|sigma - mu|^m])
4. Welch Power Spectral Density (PSD) band energies
5. 5-Fold cross-validation across 10 ML regression algorithms
6. Automatic export of CSV and JSON summary tables
"""

import os
import sys
import glob
import time
import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import welch, find_peaks

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor, BayesianRidge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold

# Ensure unbuffered standard output
sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Resolve data paths robustly
DATA_CANDIDATES = [
    os.path.join(SCRIPT_DIR, "..", "PS3", "02_Datasets", "SHM"),
    os.path.join(SCRIPT_DIR, "02_Datasets", "SHM"),
    os.path.join(SCRIPT_DIR, "data"),
    r"c:\Users\Rald999\Documents\GitHub\nebulax_p3\PS3\02_Datasets\SHM"
]

SHM_DATA_DIR = None
for cand in DATA_CANDIDATES:
    if os.path.exists(os.path.join(cand, "Train_Labels.csv")):
        SHM_DATA_DIR = os.path.abspath(cand)
        break

if SHM_DATA_DIR is None:
    raise FileNotFoundError(f"Could not locate SHM dataset in candidates: {DATA_CANDIDATES}")

TRAIN_DIR = os.path.join(SHM_DATA_DIR, "Train")
TEST_DIR = os.path.join(SHM_DATA_DIR, "Test")
LABEL_FILE = os.path.join(SHM_DATA_DIR, "Train_Labels.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "investigation_output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_turning_points(series):
    """Extract local extrema (peaks and valleys) for rainflow cycle counting."""
    diff1 = np.diff(series)
    turning = np.where(diff1[:-1] * diff1[1:] <= 0)[0] + 1
    return series[np.concatenate(([0], turning, [len(series) - 1]))]

def rainflow_fast(extrema):
    """
    O(N) stack-based 4-point rainflow cycle counting conforming to ASTM E1049-85.
    Returns:
        ranges: stress ranges (peak-to-peak delta = 2 * amplitude)
        counts: cycle count (0.5 for half cycle, 1.0 for full cycle)
    """
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

def extract_signal_features(sig):
    """Extract physics-informed and statistical features from dynamic stress time series."""
    feats = {}
    N = len(sig)
    feats['length'] = N
    
    # 1. Basic Statistical Moments
    mean_val = float(np.mean(sig))
    std_val = float(np.std(sig))
    var_val = float(np.var(sig))
    min_val = float(np.min(sig))
    max_val = float(np.max(sig))
    p2p_val = max_val - min_val
    abs_peak = float(np.max(np.abs(sig)))
    rms_val = float(np.sqrt(np.mean(sig**2)))
    mad_val = float(np.mean(np.abs(sig - mean_val)))
    
    feats['mean'] = mean_val
    feats['std'] = std_val
    feats['var'] = var_val
    feats['min'] = min_val
    feats['max'] = max_val
    feats['p2p'] = p2p_val
    feats['abs_peak'] = abs_peak
    feats['rms'] = rms_val
    feats['mad'] = mad_val
    
    # 2. Shape Factors & Percentiles
    feats['crest_factor'] = abs_peak / (rms_val + 1e-12)
    feats['kurtosis'] = float(stats.kurtosis(sig))
    feats['skewness'] = float(stats.skew(sig))
    feats['form_factor'] = rms_val / (np.mean(np.abs(sig)) + 1e-12)
    feats['impulse_factor'] = abs_peak / (np.mean(np.abs(sig)) + 1e-12)
    feats['margin_factor'] = abs_peak / ((np.mean(np.sqrt(np.abs(sig))))**2 + 1e-12)
    
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pct_vals = np.percentile(sig, pcts)
    for p, v in zip(pcts, pct_vals):
        feats[f'p{p}'] = float(v)
    feats['iqr'] = float(pct_vals[5] - pct_vals[3])  # p75 - p25
    
    zero_crossings = np.count_nonzero(np.diff(sig > 0))
    mean_crossings = np.count_nonzero(np.diff(sig > mean_val))
    feats['zero_crossing_rate'] = float(zero_crossings / N)
    feats['mean_crossing_rate'] = float(mean_crossings / N)
    
    # 3. Basquin Stress Moments: E[|sigma - mean|^m] and E[|sigma|^m]
    centered = np.abs(sig - mean_val)
    for m in [2.0, 2.5, 3.0, 3.3, 3.5, 4.0, 4.5, 5.0, 6.0]:
        feats[f'basquin_moment_{m}'] = float(np.mean(centered ** m))
        feats[f'raw_moment_{m}'] = float(np.mean(np.abs(sig) ** m))
        
    # 4. Extrema Properties
    peaks, _ = find_peaks(sig)
    valleys, _ = find_peaks(-sig)
    feats['n_peaks'] = len(peaks)
    feats['n_valleys'] = len(valleys)
    if len(peaks) > 0:
        feats['peak_mean'] = float(np.mean(sig[peaks]))
        feats['peak_std'] = float(np.std(sig[peaks]))
        feats['peak_max'] = float(np.max(sig[peaks]))
    if len(valleys) > 0:
        feats['valley_mean'] = float(np.mean(sig[valleys]))
        feats['valley_std'] = float(np.std(sig[valleys]))
        feats['valley_min'] = float(np.min(sig[valleys]))
        
    # 5. Rainflow Counting (ASTM E1049-85)
    extrema = get_turning_points(sig)
    rf_ranges, rf_counts = rainflow_fast(extrema)
    feats['rf_total_cycles'] = float(np.sum(rf_counts))
    feats['rf_max_range'] = float(np.max(rf_ranges)) if len(rf_ranges) > 0 else 0.0
    feats['rf_mean_range'] = float(np.average(rf_ranges, weights=rf_counts)) if len(rf_ranges) > 0 else 0.0
    feats['rf_rms_range'] = float(np.sqrt(np.average(rf_ranges**2, weights=rf_counts))) if len(rf_ranges) > 0 else 0.0
    
    amps = rf_ranges / 2.0
    for m in [2.0, 3.0, 3.3, 3.5, 4.0, 4.5, 5.0]:
        feats[f'rf_damage_proxy_m{m}'] = float(np.sum(rf_counts * (amps ** m)))
        
    # 6. Frequency Domain (Welch PSD)
    freqs, psd = welch(sig, fs=256.0, nperseg=2048)
    feats['psd_total_energy'] = float(np.sum(psd))
    feats['psd_peak_freq'] = float(freqs[np.argmax(psd)])
    feats['psd_mean_freq'] = float(np.sum(freqs * psd) / (np.sum(psd) + 1e-12))
    
    bands = [(0, 10), (10, 30), (30, 60), (60, 100), (100, 128)]
    for b_low, b_high in bands:
        idx = np.where((freqs >= b_low) & (freqs < b_high))[0]
        feats[f'psd_energy_{b_low}_{b_high}Hz'] = float(np.sum(psd[idx]))
        
    return feats

def main():
    print("=" * 80)
    print(" SHM DATASET COMPREHENSIVE PROFILING & BENCHMARKING ")
    print(f" Source Directory: {SHM_DATA_DIR}")
    print(f" Output Directory: {OUTPUT_DIR}")
    print("=" * 80)
    
    # 1. Label Analysis
    df_labels = pd.read_csv(LABEL_FILE)
    print(f"\n[1/6] Label Analysis (Train_Labels.csv): Shape {df_labels.shape}")
    print(df_labels['damage'].describe(percentiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]).to_string())

    # 2. Extract Features from Train Files
    train_files = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.csv")))
    print(f"\n[2/6] Extracting features from {len(train_files)} Train files...")
    train_feats = []
    t0 = time.time()
    for i, fpath in enumerate(train_files):
        fname = os.path.basename(fpath)
        arr = pd.read_csv(fpath, header=None, dtype=np.float32).values.flatten()
        f_dict = extract_signal_features(arr)
        f_dict['filename'] = fname
        train_feats.append(f_dict)
        if (i + 1) % 16 == 0 or (i + 1) == len(train_files):
            print(f"  Processed {i+1}/{len(train_files)} train files in {time.time()-t0:.1f}s")
            
    df_train_feats = pd.DataFrame(train_feats)
    df_train_full = pd.merge(df_train_feats, df_labels, on='filename')
    df_train_full.to_csv(os.path.join(OUTPUT_DIR, "train_features_with_labels.csv"), index=False)

    # 3. Extract Features from Test Files
    test_files = sorted(glob.glob(os.path.join(TEST_DIR, "*.csv")))
    print(f"\n[3/6] Extracting features from {len(test_files)} Test files...")
    test_feats = []
    t0 = time.time()
    for i, fpath in enumerate(test_files):
        fname = os.path.basename(fpath)
        arr = pd.read_csv(fpath, header=None, dtype=np.float32).values.flatten()
        f_dict = extract_signal_features(arr)
        f_dict['filename'] = fname
        test_feats.append(f_dict)
    df_test_feats = pd.DataFrame(test_feats)
    df_test_feats.to_csv(os.path.join(OUTPUT_DIR, "test_features.csv"), index=False)
    print(f"  Processed {len(test_files)} test files in {time.time()-t0:.1f}s")

    # 4. Correlation Analysis
    print(f"\n[4/6] Computing Feature Correlations with Target...")
    numeric_cols = [c for c in df_train_full.columns if c not in ['filename', 'damage', 'length']]
    log_damage = np.log(df_train_full['damage'])
    
    corr_results = []
    for col in numeric_cols:
        r_lin, _ = stats.pearsonr(df_train_full[col], df_train_full['damage'])
        s_lin, _ = stats.spearmanr(df_train_full[col], df_train_full['damage'])
        r_log, _ = stats.pearsonr(df_train_full[col], log_damage)
        r_log_log, _ = stats.pearsonr(np.log(np.maximum(df_train_full[col], 1e-12)), log_damage) if np.all(df_train_full[col] > 0) else (np.nan, np.nan)
        corr_results.append({
            'feature': col,
            'pearson_damage': float(r_lin),
            'spearman_damage': float(s_lin),
            'pearson_log_damage': float(r_log),
            'pearson_log_feat_log_damage': float(r_log_log)
        })
        
    df_corr = pd.DataFrame(corr_results)
    df_corr['abs_pearson_log_damage'] = df_corr['pearson_log_damage'].abs()
    df_corr = df_corr.sort_values(by='abs_pearson_log_damage', ascending=False)
    df_corr.to_csv(os.path.join(OUTPUT_DIR, "feature_correlations.csv"), index=False)
    
    print("\nTop 15 Most Predictive Features for log(damage):")
    print(df_corr[['feature', 'pearson_log_damage', 'pearson_damage', 'spearman_damage']].head(15).to_string(index=False))

    # 5. Domain Shift / Distribution Consistency (Train vs Test)
    print(f"\n[5/6] Checking Train vs Test Domain Consistency...")
    comparison = []
    key_features = ['mean', 'std', 'rms', 'min', 'max', 'p2p', 'abs_peak', 'kurtosis', 'skewness', 'rf_total_cycles', 'rf_damage_proxy_m3.5', 'psd_total_energy']
    for col in key_features:
        comparison.append({
            'feature': col,
            'train_mean': float(df_train_feats[col].mean()),
            'train_std': float(df_train_feats[col].std()),
            'train_min': float(df_train_feats[col].min()),
            'train_max': float(df_train_feats[col].max()),
            'test_mean': float(df_test_feats[col].mean()),
            'test_std': float(df_test_feats[col].std()),
            'test_min': float(df_test_feats[col].min()),
            'test_max': float(df_test_feats[col].max()),
        })
    df_comp = pd.DataFrame(comparison)
    df_comp.to_csv(os.path.join(OUTPUT_DIR, "train_test_comparison.csv"), index=False)
    print(df_comp[['feature', 'train_mean', 'test_mean', 'train_min', 'test_min', 'train_max', 'test_max']].to_string(index=False))

    # 6. Model Benchmarks with 5-Fold Cross Validation
    print(f"\n[6/6] Running 5-Fold CV Model Benchmarking...")
    top_12_feats = df_corr['feature'].head(12).tolist()
    X_top = df_train_full[top_12_feats].values
    y_raw = df_train_full['damage'].values
    log_y = np.log(y_raw)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    candidate_models = {
        'BayesianRidge': BayesianRidge(),
        'HuberRegressor': HuberRegressor(max_iter=500),
        'ElasticNet (alpha=0.05, l1=0.5)': ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=42),
        'Ridge (alpha=10.0)': Ridge(alpha=10.0),
        'SVR (RBF kernel, C=1.0)': SVR(C=1.0, epsilon=0.1),
        'ExtraTrees (n=100, depth=4)': ExtraTreesRegressor(n_estimators=100, max_depth=4, random_state=42),
        'RandomForest (n=100, depth=4)': RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42),
        'GradientBoosting (n=40, depth=2)': GradientBoostingRegressor(n_estimators=40, max_depth=2, random_state=42),
        'MLPRegressor (32x16)': MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42)
    }

    benchmark_rows = []
    fold_predictions = {m_name: np.zeros(len(y_raw)) for m_name in candidate_models}

    for name, model in candidate_models.items():
        mapes = []
        for tr_idx, val_idx in kf.split(X_top):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_top[tr_idx])
            X_val = scaler.transform(X_top[val_idx])
            
            model.fit(X_tr, log_y[tr_idx])
            pred_log = model.predict(X_val)
            pred = np.exp(pred_log)
            fold_predictions[name][val_idx] = pred
            
            mape = float(np.mean(np.abs(y_raw[val_idx] - pred) / y_raw[val_idx]))
            mapes.append(mape)
            
        mean_mape = float(np.mean(mapes))
        score = float(max(0, 1 - mean_mape))
        benchmark_rows.append({
            'model': name,
            'mean_mape': mean_mape,
            'competition_score': score,
            'std_mape': float(np.std(mapes))
        })

    # Test Weighted Ensemble Blend
    blend_preds = (
        0.35 * fold_predictions['BayesianRidge'] +
        0.25 * fold_predictions['ElasticNet (alpha=0.05, l1=0.5)'] +
        0.20 * fold_predictions['SVR (RBF kernel, C=1.0)'] +
        0.20 * fold_predictions['ExtraTrees (n=100, depth=4)']
    )
    blend_mape = float(np.mean(np.abs(y_raw - blend_preds) / y_raw))
    blend_score = float(max(0, 1 - blend_mape))
    benchmark_rows.append({
        'model': 'Ensemble Blend (BayesianRidge+ElasticNet+SVR+ExtraTrees)',
        'mean_mape': blend_mape,
        'competition_score': blend_score,
        'std_mape': 0.0
    })

    df_benchmarks = pd.DataFrame(benchmark_rows).sort_values(by='competition_score', ascending=False)
    df_benchmarks.to_csv(os.path.join(OUTPUT_DIR, "model_benchmark_results.csv"), index=False)
    
    print("\nBenchmark Results Summary (Ordered by Competition Score):")
    print(df_benchmarks.to_string(index=False))

    summary_report = {
        "subsystem": "SHM",
        "scoring_metric": "max(0, 1 - MAPE)",
        "train_samples": 64,
        "test_samples": 16,
        "rows_per_file": 581120,
        "benchmarks": benchmark_rows,
        "top_12_features": top_12_feats
    }
    with open(os.path.join(OUTPUT_DIR, "shm_dataset_summary.json"), "w") as f:
        json.dump(summary_report, f, indent=2)

    print(f"\n[DONE] All outputs exported to:\n  {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
