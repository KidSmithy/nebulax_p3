import os
import sys
import glob
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = r"c:\Users\Rald999\Documents\GitHub\nebulax_p3\PS3"
ACV_DIR = os.path.join(BASE_DIR, "02_Datasets", "ACV")

def check_file_quality(fpath):
    fname = os.path.basename(fpath)
    print(f"\n{'='*80}")
    print(f"DATA CLEANING AUDIT FOR: {fname}")
    print(f"{'='*80}")
    
    df = pd.read_excel(fpath)
    total_rows, total_cols = df.shape
    print(f"Total Rows: {total_rows:,} | Total Columns: {total_cols}")
    
    # 1. Timestamp checks
    time_col = None
    for c in df.columns:
        if 'time' in c.lower() or '时间' in c:
            time_col = c
            break
            
    if time_col:
        df['parsed_time'] = pd.to_datetime(df[time_col], errors='coerce')
        nat_count = df['parsed_time'].isna().sum()
        dup_count = df['parsed_time'].duplicated().sum()
        is_monotonic = df['parsed_time'].is_monotonic_increasing
        print(f"\n[Time Audit] Column: '{time_col}'")
        print(f"  - Missing/Unparseable Timestamps: {nat_count}")
        print(f"  - Duplicate Timestamps: {dup_count}")
        print(f"  - Strictly Monotonic Increasing: {is_monotonic}")
        
        # Check sampling interval deltas
        time_deltas = df['parsed_time'].diff().dt.total_seconds()
        print(f"  - Time step statistics (seconds):")
        print(f"    Min: {time_deltas.min()}s | Median: {time_deltas.median()}s | Max: {time_deltas.max()}s")
        gaps = time_deltas[time_deltas > 120]
        if len(gaps) > 0:
            print(f"    Noticeable gaps (>2 mins): {len(gaps)} instances (Max gap: {gaps.max() / 3600:.1f} hours)")

    # 2. Car Columns and Missing Values
    car_cols = [c for c in df.columns if c.startswith("Car ") and c != "Car model"]
    cars = sorted(list(set(c.split(" - ")[0].replace("Car ", "").strip() for c in car_cols)))
    
    print(f"\n[Missing Values Audit across {len(cars)} cars]:")
    missing_report = []
    for c in cars:
        c_cols = [col for col in df.columns if col.startswith(f"Car {c} - ")]
        null_counts = [df[col].isnull().sum() for col in c_cols]
        total_vals = len(df) * len(c_cols)
        sum_nulls = sum(null_counts)
        pct_null = (sum_nulls / total_vals) * 100 if total_vals > 0 else 0
        missing_report.append({
            "Car": c,
            "Param_Cols": len(c_cols),
            "Total_Nulls": sum_nulls,
            "Null_Pct": f"{pct_null:.2f}%",
            "Max_Null_In_Col": max(null_counts) if null_counts else 0
        })
    df_miss = pd.DataFrame(missing_report)
    print(df_miss.to_string(index=False))
    
    # 3. Sensor Values & Physical Outlier Audit (Temperatures)
    print(f"\n[Temperature Outlier & Anomaly Audit]:")
    # Find indoor temp cols
    temp_cols = [c for c in df.columns if any(k in c for k in ["Indoor Average Temperature", "Passenger Cabin Temperature Detected Value"])]
    
    temp_anomalies = []
    for tc in temp_cols:
        car_id = tc.split(" - ")[0].replace("Car ", "")
        s = pd.to_numeric(df[tc], errors='coerce')
        
        # Outliers: < 5°C, > 45°C, == 0.0, or sudden step jump > 10°C in 30s
        zeros = (s == 0).sum()
        negatives = (s < 0).sum()
        freezing = ((s > 0) & (s < 10)).sum()
        extreme_hot = (s > 40).sum()
        
        # Step jump
        diff_step = s.diff().abs()
        large_jumps = (diff_step > 5.0).sum()
        
        temp_anomalies.append({
            "Car": car_id,
            "Min_T": s.min(),
            "Max_T": s.max(),
            "Mean_T": np.round(s.mean(), 2),
            "Zeros": zeros,
            "Negatives": negatives,
            "Extreme_Hot (>40C)": extreme_hot,
            "Step_Jumps (>5C in 30s)": large_jumps
        })
    df_temp_anom = pd.DataFrame(temp_anomalies)
    print(df_temp_anom.to_string(index=False))
    
    # 4. Telemetry Validity Flags and Modes Audit
    print(f"\n[Validity Flags & Operating Mode Distribution]:")
    for flag_name in ["ACV Information Valid", "ACV Setting Mode", "ACV Running Mode", "Load Halved"]:
        matching_cols = [c for c in df.columns if c.endswith(f"- {flag_name}")]
        if matching_cols:
            all_vals = set()
            for mc in matching_cols:
                all_vals.update(df[mc].dropna().unique())
            print(f"  * '{flag_name}' unique values: {all_vals}")
            
            # Count invalid occurrences
            invalid_count = 0
            for mc in matching_cols:
                invalid_count += df[mc].astype(str).str.contains("Invalid|Fault|Stop", na=False).sum()
            print(f"    Total rows with Invalid/Stop across all cars: {invalid_count:,}")

def main():
    train_files = sorted(glob.glob(os.path.join(ACV_DIR, "Train", "*.xlsx")))
    test_files = sorted(glob.glob(os.path.join(ACV_DIR, "Test", "*.xlsx")))
    
    for f in train_files:
        check_file_quality(f)
    for f in test_files:
        check_file_quality(f)

if __name__ == "__main__":
    main()
