import os
import sys
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

ACV_DIR = r"c:\Users\Rald999\Documents\GitHub\nebulax_p3\PS3\02_Datasets\ACV"

def load_case_standardized(filepath):
    fname = os.path.basename(filepath)
    df = pd.read_excel(filepath)
    
    # Standardize column names
    rename_dict = {}
    for col in df.columns:
        if "Outside Temperature Sensor Reading" in col:
            rename_dict[col] = col.replace("Outside Temperature Sensor Reading", "Outdoor Average Temperature")
    df = df.rename(columns=rename_dict)
    return fname, df

def analyze_case(fname, df, faulty_car_str=None):
    print(f"\n{'='*70}")
    print(f"ANALYSIS FOR: {fname} (Ground Truth Faulty Car: {faulty_car_str})")
    print(f"{'='*70}")
    
    # Find cars
    car_cols = [c for c in df.columns if c.startswith("Car ") and c != "Car model"]
    cars = sorted(list(set(c.split(" - ")[0].replace("Car ", "").strip() for c in car_cols)))
    
    # Check if standard 8 columns exist
    first_car = cars[0]
    std_params = [
        "Indoor Average Temperature",
        "Outdoor Average Temperature",
        "ACV Control Temperature (Cooling)",
        "ACV Control Temperature (Heating)",
        "ACV Running Mode",
        "ACV Setting Mode",
        "Load Halved",
        "ACV Information Valid"
    ]
    
    # Calculate car-by-car metrics
    stats = []
    
    # Build a cross-car indoor temp dataframe
    indoor_temps = pd.DataFrame()
    cooling_setpoints = pd.DataFrame()
    running_modes = pd.DataFrame()
    load_halved = pd.DataFrame()
    
    for c in cars:
        tin_col = f"Car {c} - Indoor Average Temperature"
        tset_col = f"Car {c} - ACV Control Temperature (Cooling)"
        rmode_col = f"Car {c} - ACV Running Mode"
        lh_col = f"Car {c} - Load Halved"
        
        if tin_col in df.columns:
            indoor_temps[c] = pd.to_numeric(df[tin_col], errors='coerce')
        if tset_col in df.columns:
            cooling_setpoints[c] = pd.to_numeric(df[tset_col], errors='coerce')
        if rmode_col in df.columns:
            running_modes[c] = pd.to_numeric(df[rmode_col], errors='coerce')
        if lh_col in df.columns:
            load_halved[c] = pd.to_numeric(df[lh_col], errors='coerce')

    if not indoor_temps.empty:
        # Cross-car median at each timestamp
        median_tin = indoor_temps.median(axis=1)
        mean_tin = indoor_temps.mean(axis=1)
        
        for c in cars:
            t_series = indoor_temps[c]
            # Relative temp diff vs median of train
            rel_diff = t_series - median_tin
            
            # Temp diff vs cooling setpoint (when cooling is active)
            t_set = cooling_setpoints[c] if c in cooling_setpoints.columns else None
            diff_set = (t_series - t_set) if t_set is not None else None
            
            # Load halved rate
            lh_rate = load_halved[c].mean() if c in load_halved.columns else np.nan
            
            # Running mode distribution
            rm_val_counts = running_modes[c].value_counts().to_dict() if c in running_modes.columns else {}
            
            is_fault = (c == faulty_car_str)
            
            stats.append({
                "Car": c,
                "Is_Fault": is_fault,
                "Mean_Indoor_T": np.round(t_series.mean(), 2),
                "Max_Indoor_T": np.round(t_series.max(), 2),
                "Mean_Rel_Diff": np.round(rel_diff.mean(), 3),
                "Median_Rel_Diff": np.round(rel_diff.median(), 3),
                "95th_Rel_Diff": np.round(rel_diff.quantile(0.95), 3),
                "Max_Rel_Diff": np.round(rel_diff.max(), 3),
                "Mean_T_minus_Set": np.round(diff_set.mean(), 3) if diff_set is not None else np.nan,
                "Load_Halved_Rate": np.round(lh_rate, 4),
                "Running_Modes": rm_val_counts
            })
            
        df_stats = pd.DataFrame(stats)
        print("\nCar Comparison Table:")
        cols_to_show = ["Car", "Is_Fault", "Mean_Indoor_T", "Mean_Rel_Diff", "Median_Rel_Diff", "95th_Rel_Diff", "Max_Rel_Diff", "Mean_T_minus_Set", "Load_Halved_Rate"]
        print(df_stats[cols_to_show].to_string(index=False))
        
        # Check rank of the faulty car if ranked by Mean_Rel_Diff (descending)
        ranked_by_rel = df_stats.sort_values(by="Mean_Rel_Diff", ascending=False)["Car"].tolist()
        print(f"\nRanking by Mean Relative Temp Diff (Highest first): {ranked_by_rel}")
        if faulty_car_str:
            rank = ranked_by_rel.index(faulty_car_str) + 1
            score = (len(cars) - (rank - 1)) / len(cars)
            print(f"-> Faulty car {faulty_car_str} Rank: {rank}/{len(cars)} | Score: {score:.4f}")
            
        # Check rank by Mean T minus Set
        if "Mean_T_minus_Set" in df_stats.columns:
            ranked_by_tset = df_stats.sort_values(by="Mean_T_minus_Set", ascending=False)["Car"].tolist()
            print(f"Ranking by Mean (Indoor - Setpoint): {ranked_by_tset}")
            if faulty_car_str:
                rank_set = ranked_by_tset.index(faulty_car_str) + 1
                score_set = (len(cars) - (rank_set - 1)) / len(cars)
                print(f"-> Faulty car {faulty_car_str} Rank: {rank_set}/{len(cars)} | Score: {score_set:.4f}")
                
        # Running mode details
        print("\nRunning Modes Summary across cars:")
        for s in stats:
            print(f"  Car {s['Car']} (Fault: {s['Is_Fault']}): {s['Running_Modes']}")

def main():
    labels = {
        "acv_case_01.xlsx": "01",
        "acv_case_02.xlsx": "02",
        "acv_case_03.xlsx": "03",
        "acv_case_05.xlsx": "04",
        "acv_case_06.xlsx": "06",
        "acv_test_case.xlsx": None
    }
    
    for fname, faulty_car in labels.items():
        if fname.startswith("acv_test"):
            fpath = os.path.join(ACV_DIR, "Test", fname)
        else:
            fpath = os.path.join(ACV_DIR, "Train", fname)
        fname_loaded, df = load_case_standardized(fpath)
        analyze_case(fname_loaded, df, faulty_car)

if __name__ == "__main__":
    main()
