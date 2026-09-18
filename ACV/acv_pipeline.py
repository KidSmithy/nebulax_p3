"""
acv_pipeline.py
=============================================================================
Production Inference Interface for ACV Refrigerant Leakage Diagnosis.
Designed for UI (Streamlit, Flask, FastAPI) and automated batch evaluation.
=============================================================================
"""

import os
import io
import re
import joblib
import numpy as np
import pandas as pd

DEFAULT_BUNDLE_PATH = os.path.join(os.path.dirname(__file__), "acv_model_bundle.joblib")
if not os.path.exists(DEFAULT_BUNDLE_PATH):
    alt_path = os.path.join(os.path.dirname(__file__), "..", "PS3", "models", "acv_model_bundle.joblib")
    if os.path.exists(alt_path):
        DEFAULT_BUNDLE_PATH = alt_path

class ACVPredictor:
    """Production Inference Engine for Train ACV Refrigerant Leakage Diagnosis."""
    
    def __init__(self, bundle_path: str = None):
        if bundle_path is None:
            bundle_path = DEFAULT_BUNDLE_PATH
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"ACV Model bundle not found at {bundle_path}. Run export_acv_model_bundle.py first.")
            
        self.bundle = joblib.load(bundle_path)
        self.version = self.bundle.get('version', '1.0.0')
        self.pairwise_model = self.bundle.get('pairwise_model', None)
        self.diff_cols = self.bundle.get('diff_cols', [])
        self.physics_weights = self.bundle.get('physics_weights', {})
        self.ensemble_weights = self.bundle.get('ensemble_weights', {'physics_weight': 0.75, 'pairwise_linear_weight': 0.25})
        self.cleaning_thresholds = self.bundle.get('cleaning_thresholds', {'temp_min': 5.0, 'temp_max': 45.0})

    def clean_telemetry(self, raw_df: pd.DataFrame):
        df = raw_df.copy()
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
        
        car_cols = [c for c in df.columns if c.startswith("Car ") and c != "Car model"]
        all_cids = sorted(list(set(c.split(" - ")[0].replace("Car ", "").strip() for c in car_cols)))
        active_cars = [cid for cid in all_cids if f"Car {cid} - Indoor Average Temperature" in df.columns 
                       and df[f"Car {cid} - Indoor Average Temperature"].notnull().sum() > 0]
                       
        t_min = self.cleaning_thresholds.get('temp_min', 5.0)
        t_max = self.cleaning_thresholds.get('temp_max', 45.0)
        
        for cid in active_cars:
            t_col = f"Car {cid} - Indoor Average Temperature"
            valid_col = f"Car {cid} - ACV Information Valid"
            if valid_col in df.columns:
                invalid_mask = df[valid_col].astype(str).str.contains("Invalid", na=False)
                df.loc[invalid_mask, t_col] = np.nan
            if t_col in df.columns:
                s = pd.to_numeric(df[t_col], errors='coerce')
                s = s.mask((s <= t_min) | (s >= t_max), np.nan)
                df[t_col] = s.ffill().bfill()
                
        active_temp_cols = [f"Car {cid} - Indoor Average Temperature" for cid in active_cars if f"Car {cid} - Indoor Average Temperature" in df.columns]
        df = df.dropna(subset=active_temp_cols, how='all').reset_index(drop=True)
        return df, active_cars

    def extract_features(self, df: pd.DataFrame, active_cars: list):
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
            
            rolling_elevated = (rel_diff.rolling(window=30, min_periods=10).mean() > 0.20)
            persistence_frac = float(rolling_elevated[is_cooling].mean()) if len(rolling_elevated[is_cooling].dropna()) > 0 else 0.0
            
            features[c] = {
                "mean_indoor": float(np.nanmean(tin)),
                "mean_rel_diff_cooling": float(np.nanmean(rel_cooling)) if len(rel_cooling.dropna()) > 0 else float(np.nanmean(rel_diff)),
                "mean_t_minus_set_cooling": float(np.nanmean(t_minus_set_cooling)) if len(t_minus_set_cooling.dropna()) > 0 else 0.0,
                "p95_rel_diff": float(np.nanpercentile(rel_diff.dropna(), 95)) if len(rel_diff.dropna()) > 0 else 0.0,
                "frac_above_median": float((rel_diff > 0.20).mean()),
                "persistence_15m": persistence_frac
            }
            
        df_feat = pd.DataFrame(features).T
        df_feat_z = df_feat.copy()
        raw_cols = ["mean_rel_diff_cooling", "mean_t_minus_set_cooling", "p95_rel_diff", "frac_above_median", "persistence_15m"]
        for col in raw_cols:
            std_val = df_feat[col].std()
            df_feat_z[f"{col}_z"] = (df_feat[col] - df_feat[col].mean()) / (std_val if std_val > 1e-6 else 1.0)
            
        return df_feat, df_feat_z

    def predict(self, df_or_filepath) -> dict:
        """
        Accepts a filepath (.xlsx/.csv) or a pandas DataFrame.
        Returns a rich prediction dictionary for frontend UI consumption.
        """
        if isinstance(df_or_filepath, str):
            if df_or_filepath.endswith('.csv'):
                df = pd.read_csv(df_or_filepath)
            else:
                df = pd.read_excel(df_or_filepath)
        elif isinstance(df_or_filepath, (io.BytesIO, io.StringIO)):
            try:
                df = pd.read_excel(df_or_filepath)
            except Exception:
                df_or_filepath.seek(0)
                df = pd.read_csv(df_or_filepath)
        elif isinstance(df_or_filepath, pd.DataFrame):
            df = df_or_filepath
        else:
            raise ValueError("Input must be a filepath string, BytesIO buffer, or pandas DataFrame.")
            
        clean_df, active_cars = self.clean_telemetry(df)
        df_feat, df_feat_z = self.extract_features(clean_df, active_cars)
        
        # 1. Physics Score
        w = self.physics_weights
        phys_scores = {}
        for c in active_cars:
            s = (w.get('w1_rel_diff', 1.0) * df_feat_z.loc[c, "mean_rel_diff_cooling_z"] +
                 w.get('w2_setpoint_error', 0.5) * df_feat_z.loc[c, "mean_t_minus_set_cooling_z"] +
                 w.get('w3_p95_delta', 0.25) * df_feat_z.loc[c, "p95_rel_diff_z"] +
                 w.get('w4_frac_above_median', 0.25) * df_feat_z.loc[c, "frac_above_median_z"] +
                 w.get('w5_persistence', 0.25) * df_feat_z.loc[c, "persistence_15m_z"])
            phys_scores[c] = s
            
        s_phys = pd.Series(phys_scores)
        std_p = s_phys.std()
        phys_z = (s_phys - s_phys.mean()) / (std_p if std_p > 1e-6 else 1.0)
        
        # 2. Pairwise Score
        pairwise_scores = {}
        feature_keys = ["mean_rel_diff_cooling_z", "mean_t_minus_set_cooling_z", "p95_rel_diff_z", "frac_above_median_z", "persistence_15m_z"]
        for ca in active_cars:
            win_probs = []
            for cb in active_cars:
                if ca == cb:
                    continue
                diff_dict = {f"diff_{k}": [df_feat_z.loc[ca, k] - df_feat_z.loc[cb, k]] for k in feature_keys}
                df_diff = pd.DataFrame(diff_dict)[self.diff_cols]
                p_win = self.pairwise_model.predict_proba(df_diff)[0, 1]
                win_probs.append(p_win)
            pairwise_scores[ca] = np.mean(win_probs)
            
        s_pw = pd.Series(pairwise_scores)
        std_pw = s_pw.std()
        pw_z = (s_pw - s_pw.mean()) / (std_pw if std_pw > 1e-6 else 1.0)
        
        # 3. Ensemble Score
        w_phys = self.ensemble_weights.get('physics_weight', 0.75)
        w_pw = self.ensemble_weights.get('pairwise_linear_weight', 0.25)
        final_scores = w_phys * phys_z + w_pw * pw_z
        
        sorted_cars = final_scores.sort_values(ascending=False).index.tolist()
        ranked_cars_str = "|".join(sorted_cars)
        top_car = sorted_cars[0]
        
        if len(sorted_cars) > 1:
            margin = round(float(final_scores[top_car] - final_scores[sorted_cars[1]]), 3)
        else:
            margin = 1.0
            
        car_diag = []
        for r_idx, c in enumerate(sorted_cars, 1):
            car_diag.append({
                'rank': r_idx,
                'car': c,
                'final_score': round(float(final_scores[c]), 3),
                'physics_score_z': round(float(phys_z[c]), 3),
                'pairwise_prob': round(float(pairwise_scores[c]), 3),
                'mean_delta_rel': round(float(df_feat.loc[c, 'mean_rel_diff_cooling']), 2),
                'mean_delta_set': round(float(df_feat.loc[c, 'mean_t_minus_set_cooling']), 2),
                'p95_delta_rel': round(float(df_feat.loc[c, 'p95_rel_diff']), 2),
                'persistence_pct': round(float(df_feat.loc[c, 'persistence_15m']) * 100, 1)
            })
            
        summary = (
            f"Car {top_car} diagnosed as having refrigerant leakage with score margin {margin}. "
            f"Observed mean temperature deficit of +{df_feat.loc[top_car, 'mean_rel_diff_cooling']:.2f}°C relative to consist median, "
            f"and +{df_feat.loc[top_car, 'mean_t_minus_set_cooling']:.2f}°C setpoint error during active cooling."
        )
        
        return {
            'ranked_cars': ranked_cars_str,
            'ranked_cars_list': sorted_cars,
            'most_likely_faulty_car': top_car,
            'confidence_margin': margin,
            'status': f"Refrigerant Leakage Fault Diagnosed in Car {top_car}",
            'consist_size': len(sorted_cars),
            'active_cars': active_cars,
            'car_diagnostics': car_diag,
            'diagnostic_summary': summary
        }

    def predict_file(self, file_path: str) -> dict:
        return self.predict(file_path)

if __name__ == "__main__":
    test_excel = os.path.join(os.path.dirname(__file__), "..", "PS3", "02_Datasets", "ACV", "acv_test_case.xlsx")
    if not os.path.exists(test_excel):
        test_excel = os.path.join(os.path.dirname(__file__), "acv_test_case.xlsx")
    
    print(f"Self-testing ACVPredictor on: {test_excel}")
    if os.path.exists(test_excel):
        predictor = ACVPredictor()
        res = predictor.predict(test_excel)
        print("\n--- Diagnostic Prediction Result ---")
        print(f"Ranked Cars String : {res['ranked_cars']}")
        print(f"Top Faulty Car     : Car {res['most_likely_faulty_car']}")
        print(f"Confidence Margin  : {res['confidence_margin']}")
        print(f"Summary            : {res['diagnostic_summary']}")
        print("\nPer-Car Diagnostics Table:")
        print(pd.DataFrame(res['car_diagnostics'])[['rank', 'car', 'final_score', 'mean_delta_rel', 'mean_delta_set', 'persistence_pct']])
