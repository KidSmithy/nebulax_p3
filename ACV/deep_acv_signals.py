import os
import sys
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

ACV_DIR = r"c:\Users\Rald999\Documents\GitHub\nebulax_p3\PS3\02_Datasets\ACV"

def inspect_pressures_case_04():
    print("\n" + "="*80)
    print("PHYSICAL CONFIRMATION: REFRIGERANT PRESSURE IN CASE 04 (Fault is Car 01)")
    print("="*80)
    df4 = pd.read_excel(os.path.join(ACV_DIR, "Train", "acv_case_04.xlsx"))
    
    active_cars = ['01', '02', '03', '04']
    for p_name in [
        "Refrigeration System 1 Low Pressure Value",
        "Refrigeration System 1 High Pressure Value",
        "Refrigeration System 2 Low Pressure Value",
        "Refrigeration System 2 High Pressure Value"
    ]:
        print(f"\nParameter: {p_name}")
        for c in active_cars:
            col = f"Car {c} - {p_name}"
            s = pd.to_numeric(df4[col], errors='coerce')
            print(f"  Car {c} (Fault={c=='01'}): Mean={s.mean():.2f}, Min={s.min():.2f}, Max={s.max():.2f}, 10th={s.quantile(0.1):.2f}, 50th={s.median():.2f}")

def inspect_test_case_deep():
    print("\n" + "="*80)
    print("DEEP ANALYSIS OF TEST CASE: acv_test_case.xlsx")
    print("="*80)
    df = pd.read_excel(os.path.join(ACV_DIR, "Test", "acv_test_case.xlsx"))
    cars = ['01', '02', '03', '04', '05', '06', '07', '08']
    
    # Check Car model, Train number, Time
    print(f"Metadata: Car model={df['Car model'].unique()}, Train number={df['Train number'].unique()}")
    print(f"Time range: {df['Time'].min()} to {df['Time'].max()}, total rows={len(df)}")
    
    # Check Outdoor temperature
    out_cols = [f"Car {c} - Outdoor Average Temperature" for c in cars if f"Car {c} - Outdoor Average Temperature" in df.columns]
    if out_cols:
        df_out = df[out_cols].apply(pd.to_numeric, errors='coerce')
        print(f"Outdoor temperature: Min={df_out.min().min():.1f}, Mean={df_out.mean().mean():.1f}, Max={df_out.max().max():.1f}")
        
    # Check Running Modes
    print("\nRunning Modes across cars in Test:")
    for c in cars:
        col = f"Car {c} - ACV Running Mode"
        if col in df.columns:
            vc = df[col].value_counts().to_dict()
            print(f"  Car {c}: {vc}")
            
    # Check Load Halved
    print("\nLoad Halved across cars in Test:")
    for c in cars:
        col = f"Car {c} - Load Halved"
        if col in df.columns:
            vc = df[col].value_counts().to_dict()
            print(f"  Car {c}: {vc}")

    # Cross-car indoor temp
    indoor_df = pd.DataFrame({c: pd.to_numeric(df[f"Car {c} - Indoor Average Temperature"], errors='coerce') for c in cars})
    cooling_set_df = pd.DataFrame({c: pd.to_numeric(df[f"Car {c} - ACV Control Temperature (Cooling)"], errors='coerce') for c in cars})
    
    # Filter only when ACV is running / cooling
    median_indoor = indoor_df.median(axis=1)
    
    test_car_stats = []
    for c in cars:
        t_in = indoor_df[c]
        t_set = cooling_set_df[c]
        rel_diff = t_in - median_indoor
        diff_set = t_in - t_set
        
        # When train is cooling (e.g. median indoor temp > 22 and running mode contains 'Cooling')
        mode_col = f"Car {c} - ACV Running Mode"
        is_cooling = df[mode_col].str.contains("Cooling", na=False) if mode_col in df.columns else pd.Series(True, index=df.index)
        
        rel_diff_cooling = rel_diff[is_cooling]
        diff_set_cooling = diff_set[is_cooling]
        
        test_car_stats.append({
            "Car": c,
            "Mean_Indoor": np.round(t_in.mean(), 3),
            "Max_Indoor": np.round(t_in.max(), 3),
            "Mean_Rel_Diff": np.round(rel_diff.mean(), 4),
            "Median_Rel_Diff": np.round(rel_diff.median(), 4),
            "75th_Rel_Diff": np.round(rel_diff.quantile(0.75), 4),
            "90th_Rel_Diff": np.round(rel_diff.quantile(0.90), 4),
            "95th_Rel_Diff": np.round(rel_diff.quantile(0.95), 4),
            "Max_Rel_Diff": np.round(rel_diff.max(), 4),
            "Mean_Rel_Diff_Cooling": np.round(rel_diff_cooling.mean(), 4),
            "Mean_Diff_Set_Cooling": np.round(diff_set_cooling.mean(), 4)
        })
        
    df_test_stats = pd.DataFrame(test_car_stats)
    print("\nTest Case Car Comparison Table:")
    print(df_test_stats.to_string(index=False))
    
    print("\nRankings on Test Case:")
    for col in ["Mean_Rel_Diff", "Mean_Rel_Diff_Cooling", "Mean_Diff_Set_Cooling", "95th_Rel_Diff"]:
        ranked = df_test_stats.sort_values(by=col, ascending=False)["Car"].tolist()
        print(f"  Ranked by {col:<24}: {' | '.join(ranked)}")

if __name__ == "__main__":
    inspect_pressures_case_04()
    inspect_test_case_deep()
