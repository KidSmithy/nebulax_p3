"""
SHM Subsystem Prediction Script: Rail Vehicle Fatigue Damage Assessment
=======================================================================
Complies with official CLI interface:
  python predict.py --input <path_to_test_directory> --output <path_to_output_csv>

Supports:
1. Fast inference using pre-trained model artifact (shm_model.pkl) if available.
2. Direct fallback to training if artifact is missing.
"""

import os
import glob
import pickle
import argparse
import numpy as np
import pandas as pd
from scipy.signal import welch

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
    # CRITICAL: header=None because data files have no column headers!
    sig = pd.read_csv(filepath, header=None, dtype=np.float32).values.flatten()
    
    mean_val = float(np.mean(sig))
    min_val = float(np.min(sig))
    max_val = float(np.max(sig))
    p2p_val = max_val - min_val
    abs_peak = float(np.max(np.abs(sig)))
    mad_val = float(np.mean(np.abs(sig - mean_val)))
    
    pct_vals = np.percentile(sig, [25, 75])
    iqr_val = float(pct_vals[1] - pct_vals[0])
    
    # Rainflow Miner's rule proxies
    extrema = get_turning_points(sig)
    rf_ranges, rf_counts = rainflow_fast(extrema)
    amps = rf_ranges / 2.0
    
    rf_proxy_33 = float(np.sum(rf_counts * (amps ** 3.3)))
    rf_proxy_35 = float(np.sum(rf_counts * (amps ** 3.5)))
    rf_proxy_40 = float(np.sum(rf_counts * (amps ** 4.0)))
    rf_proxy_45 = float(np.sum(rf_counts * (amps ** 4.5)))
    
    # Welch PSD
    freqs, psd = welch(sig, fs=256.0, nperseg=2048)
    psd_total = float(np.sum(psd))
    psd_0_10 = float(np.sum(psd[np.where((freqs >= 0) & (freqs < 10))[0]]))
    
    return [
        p2p_val, abs_peak, min_val, min_val, psd_total, mad_val,
        rf_proxy_35, psd_0_10, rf_proxy_40, rf_proxy_33, iqr_val, rf_proxy_45
    ]

def predict_with_model(model_path, test_dir, output_csv):
    print(f"Loading trained model artifact from: {model_path}")
    with open(model_path, 'rb') as f:
        payload = pickle.load(f)

    scaler = payload['scaler']
    models = payload['models']
    weights = payload['weights']
    clip_min, clip_max = payload.get('clip_range', (0.025, 0.950))

    test_files = sorted(glob.glob(os.path.join(test_dir, "*.csv")))
    print(f"Found {len(test_files)} test files in {test_dir}. Generating predictions...")
    
    predictions = []
    for f in test_files:
        fname = os.path.basename(f)
        feats = np.array([extract_file_features(f)])
        feats_scaled = scaler.transform(feats)

        preds = [np.exp(m.predict(feats_scaled)[0]) for m in models]
        p_blend = sum(w * p for w, p in zip(weights, preds))
        p_blend = float(np.clip(p_blend, clip_min, clip_max))

        predictions.append({'file_id': fname, 'prediction': p_blend})

    df_out = pd.DataFrame(predictions)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f"[SUCCESS] Predictions written to: {output_csv}")
    print(df_out.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict SHM Cumulative Fatigue Damage")
    parser.add_argument("--input", required=True, help="Path to input test directory")
    parser.add_argument("--output", required=True, help="Path to output prediction CSV")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "shm_model.pkl")

    if not os.path.exists(model_path):
        print("Pre-trained model not found. Running train.py first...")
        from train import main as run_training
        run_training()

    predict_with_model(model_path, args.input, args.output)
