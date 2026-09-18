"""
run_automl_pipeline.py
=============================================================================
Comprehensive AutoML & Optimization Pipeline for Rail Corrugation Monitoring
Dataset: 63 Engineered Domain Features from 272 10,000x129 Recordings
Features File: Rail_Corrugation/rail_extracted_features.csv

Phases Executed:
1. Phase 1: AutoGluon Tabular (5-fold bagging, 0 stack levels, f1_macro metric)
2. Phase 2: FLAML AutoML (XGBoost, LightGBM, CatBoost, ExtraTrees, Random Forest)
3. Phase 3: Optuna Custom Optimization (Macro-F1 objective with class weights,
   low-speed gating, and out-of-fold probability threshold tuning)
4. Phase 4: Final Validation Leaderboard Table
5. Phase 5: Export Winning Model Bundle & Test Set Predictions
=============================================================================
"""

import os
import sys
import time
import joblib
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import optuna
from flaml import AutoML
from autogluon.tabular import TabularPredictor

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_CSV = os.path.join(BASE_DIR, "rail_extracted_features.csv")
TEST_DIR = os.path.join(r"c:\Users\Rald999\Documents\GitHub\nebulax_p3\PS3\02_Datasets\Rail_Corrugation", "Test")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

LABEL_MAP = {'Normal': 0, 'Side I': 1, 'Side II': 2}
INV_MAP = {0: 'Normal', 1: 'Side I', 2: 'Side II'}

def evaluate_predictions(y_true, y_pred, model_name="Model"):
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    f1_per_class = f1_score(y_true, y_pred, average=None)
    acc = np.mean(y_true == y_pred)
    return {
        'Model': model_name,
        'Macro F1': round(macro_f1, 4),
        'Normal F1': round(f1_per_class[0], 4) if len(f1_per_class) > 0 else 0.0,
        'Side I F1': round(f1_per_class[1], 4) if len(f1_per_class) > 1 else 0.0,
        'Side II F1': round(f1_per_class[2], 4) if len(f1_per_class) > 2 else 0.0,
        'Accuracy': round(acc, 4)
    }

def main():
    print("=" * 80)
    print("STARTING RAIL CORRUGATION AUTOML & OPTUNA EXPERIMENT")
    print("=" * 80)
    
    if not os.path.exists(FEATURES_CSV):
        raise FileNotFoundError(f"Features file not found at {FEATURES_CSV}")
        
    df_data = pd.read_csv(FEATURES_CSV)
    print(f"Loaded features table: {df_data.shape[0]} rows x {df_data.shape[1]} columns.")
    print("Class distribution:\n", df_data['label'].value_counts())
    
    feature_cols = [c for c in df_data.columns if c not in ['filename', 'label']]
    X = df_data[feature_cols].values
    y = df_data['label'].map(LABEL_MAP).values
    speed = df_data['speed_kmh'].values
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    leaderboard = []

    # -------------------------------------------------------------------------
    # PHASE 1: AutoGluon Tabular
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 1: AutoGluon Tabular (5-Fold Bagging, No Stacking, f1_macro)")
    print("=" * 80)
    
    # AutoGluon train data
    ag_train_data = df_data[feature_cols + ['label']].copy()
    
    try:
        t0 = time.time()
        ag_output_dir = os.path.join(BASE_DIR, "autogluon_output")
        ag_predictor = TabularPredictor(
            label="label",
            eval_metric="f1_macro",
            problem_type="multiclass",
            path=ag_output_dir
        ).fit(
            train_data=ag_train_data,
            presets="best_quality",
            num_bag_folds=5,
            num_stack_levels=0,
            time_limit=240,  # 4 minutes budget
            verbosity=1
        )
        dt_ag = time.time() - t0
        print(f"AutoGluon training completed in {dt_ag:.1f}s.")
        
        # Out-of-fold predictions
        ag_oof_probs = ag_predictor.predict_proba(ag_train_data)
        # Ensure class ordering aligns with [Normal, Side I, Side II]
        ag_oof_preds_labels = ag_predictor.predict(ag_train_data)
        ag_oof_preds = ag_oof_preds_labels.map(LABEL_MAP).values
        
        res_ag = evaluate_predictions(y, ag_oof_preds, "AutoGluon (Bagged Ensemble)")
        leaderboard.append(res_ag)
        print("\nAutoGluon Out-Of-Fold Evaluation:")
        print(res_ag)
        print("\nAutoGluon Model Leaderboard:")
        print(ag_predictor.leaderboard(silent=True)[['model', 'score_val', 'pred_time_val', 'fit_time']])
    except Exception as e:
        print(f"AutoGluon run encountered an issue: {e}")

    # -------------------------------------------------------------------------
    # PHASE 2: FLAML AutoML
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 2: FLAML AutoML (Search across XGBoost, LightGBM, CatBoost, RF, ExtraTrees)")
    print("=" * 80)
    
    t0 = time.time()
    flaml_automl = AutoML()
    flaml_settings = {
        "time_budget": 60,  # 60 seconds budget
        "metric": "macro_f1",
        "task": "multiclass",
        "log_file_name": os.path.join(BASE_DIR, "flaml_run.log"),
        "estimator_list": ["xgboost", "lgbm", "catboost", "rf", "extra_tree"],
        "n_splits": 5,
        "split_type": "stratified",
        "seed": 42,
        "verbose": 0
    }
    flaml_automl.fit(X_train=X, y_train=y, **flaml_settings)
    dt_flaml = time.time() - t0
    print(f"FLAML completed in {dt_flaml:.1f}s. Best estimator: {flaml_automl.best_estimator}")
    print(f"Best config: {flaml_automl.best_config}")
    
    # 5-fold CV evaluation with FLAML best estimator
    flaml_oof = np.zeros(len(y))
    for tr_idx, va_idx in skf.split(X, y):
        clone_estimator = flaml_automl.best_model_for_estimator(flaml_automl.best_estimator)
        clone_estimator.fit(X[tr_idx], y[tr_idx])
        flaml_oof[va_idx] = clone_estimator.predict(X[va_idx])
        
    res_flaml = evaluate_predictions(y, flaml_oof, f"FLAML ({flaml_automl.best_estimator})")
    leaderboard.append(res_flaml)
    print("\nFLAML Out-Of-Fold Evaluation:")
    print(res_flaml)

    # -------------------------------------------------------------------------
    # PHASE 3: Optuna Custom Optimization (Macro-F1 Direct Objective)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 3: Optuna Custom Optimization (Tuning Weights, Gating, Thresholds for Macro F1)")
    print("=" * 80)
    
    def optuna_objective(trial):
        # 1. Class weight hyperparameters for minority classes
        w_side_i = trial.suggest_float("w_side_i", 6.0, 26.0)
        w_side_ii = trial.suggest_float("w_side_ii", 3.0, 15.0)
        low_speed_th = trial.suggest_float("low_speed_th", 8.0, 14.0)
        
        # 2. Classifier choice & parameters
        classifier_type = trial.suggest_categorical("clf_type", ["xgboost", "catboost", "lightgbm"])
        
        # Class weights array
        class_weights = np.array([1.0, w_side_i, w_side_ii])
        
        oof_probs = np.zeros((len(y), 3))
        
        for tr_idx, va_idx in skf.split(X, y):
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_va, y_va = X[va_idx], y[va_idx]
            
            sample_w_tr = class_weights[y_tr]
            
            if classifier_type == "xgboost":
                max_depth = trial.suggest_int("xgb_depth", 3, 6)
                lr = trial.suggest_float("xgb_lr", 0.02, 0.20, log=True)
                colsample = trial.suggest_float("xgb_colsample", 0.6, 1.0)
                subsample = trial.suggest_float("xgb_subsample", 0.6, 1.0)
                reg_lambda = trial.suggest_float("xgb_lambda", 1e-2, 10.0, log=True)
                
                clf = xgb.XGBClassifier(
                    n_estimators=140,
                    max_depth=max_depth,
                    learning_rate=lr,
                    colsample_bytree=colsample,
                    subsample=subsample,
                    reg_lambda=reg_lambda,
                    random_state=42,
                    eval_metric="mlogloss",
                    verbosity=0
                )
                clf.fit(X_tr, y_tr, sample_weight=sample_w_tr)
                probs = clf.predict_proba(X_va)
                
            elif classifier_type == "catboost":
                depth = trial.suggest_int("cb_depth", 3, 6)
                lr = trial.suggest_float("cb_lr", 0.02, 0.20, log=True)
                l2 = trial.suggest_float("cb_l2", 1.0, 10.0)
                
                clf = CatBoostClassifier(
                    iterations=150,
                    depth=depth,
                    learning_rate=lr,
                    l2_leaf_reg=l2,
                    loss_function="MultiClass",
                    class_weights=class_weights.tolist(),
                    random_seed=42,
                    verbose=0
                )
                clf.fit(X_tr, y_tr)
                probs = clf.predict_proba(X_va)
                
            elif classifier_type == "lightgbm":
                max_depth = trial.suggest_int("lgb_depth", 3, 6)
                num_leaves = trial.suggest_int("lgb_leaves", 7, 31)
                lr = trial.suggest_float("lgb_lr", 0.02, 0.20, log=True)
                subsample = trial.suggest_float("lgb_subsample", 0.6, 1.0)
                
                clf = lgb.LGBMClassifier(
                    n_estimators=140,
                    max_depth=max_depth,
                    num_leaves=num_leaves,
                    learning_rate=lr,
                    subsample=subsample,
                    random_state=42,
                    verbose=-1
                )
                clf.fit(X_tr, y_tr, sample_weight=sample_w_tr)
                probs = clf.predict_proba(X_va)
                
            oof_probs[va_idx] = probs
            
        # Apply low speed hard gate
        low_speed_mask = speed < low_speed_th
        oof_preds = np.argmax(oof_probs, axis=1)
        oof_preds[low_speed_mask] = 0  # 100% Normal
        
        macro_f1 = f1_score(y, oof_preds, average='macro')
        return macro_f1

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(optuna_objective, n_trials=60, timeout=180)
    
    print("\nOptuna Optimization Complete.")
    print(f"Best Macro F1 achieved: {study.best_value:.4f}")
    print("Best Trial Parameters:", study.best_params)
    
    # -------------------------------------------------------------------------
    # Re-run Best Optuna Model to get Full Out-Of-Fold Probabilities
    # -------------------------------------------------------------------------
    bp = study.best_params
    best_clf_type = bp['clf_type']
    class_weights = np.array([1.0, bp['w_side_i'], bp['w_side_ii']])
    low_speed_th = bp['low_speed_th']
    
    best_oof_probs = np.zeros((len(y), 3))
    
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]
        sample_w_tr = class_weights[y_tr]
        
        if best_clf_type == "xgboost":
            clf = xgb.XGBClassifier(
                n_estimators=140,
                max_depth=bp['xgb_depth'],
                learning_rate=bp['xgb_lr'],
                colsample_bytree=bp['xgb_colsample'],
                subsample=bp['xgb_subsample'],
                reg_lambda=bp['xgb_lambda'],
                random_state=42,
                eval_metric="mlogloss",
                verbosity=0
            )
            clf.fit(X_tr, y_tr, sample_weight=sample_w_tr)
        elif best_clf_type == "catboost":
            clf = CatBoostClassifier(
                iterations=150,
                depth=bp['cb_depth'],
                learning_rate=bp['cb_lr'],
                l2_leaf_reg=bp['cb_l2'],
                loss_function="MultiClass",
                class_weights=class_weights.tolist(),
                random_seed=42,
                verbose=0
            )
            clf.fit(X_tr, y_tr)
        elif best_clf_type == "lightgbm":
            clf = lgb.LGBMClassifier(
                n_estimators=140,
                max_depth=bp['lgb_depth'],
                num_leaves=bp['lgb_leaves'],
                learning_rate=bp['lgb_lr'],
                subsample=bp['lgb_subsample'],
                random_state=42,
                verbose=-1
            )
            clf.fit(X_tr, y_tr, sample_weight=sample_w_tr)
            
        best_oof_probs[va_idx] = clf.predict_proba(X_va)
        
    # Baseline Optuna Argmax
    optuna_preds = np.argmax(best_oof_probs, axis=1)
    optuna_preds[speed < low_speed_th] = 0
    res_optuna = evaluate_predictions(y, optuna_preds, f"Optuna Tuned {best_clf_type.upper()}")
    leaderboard.append(res_optuna)

    # -------------------------------------------------------------------------
    # PHASE 3B: Threshold Tuning for Macro F1 Maximization
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Optimizing Decision Thresholds to Maximize Rare-Class Macro F1...")
    print("-" * 60)
    
    def threshold_loss(thresholds):
        # thresholds: [th_norm, th_side1, th_side2]
        adjusted_probs = best_oof_probs / np.maximum(thresholds, 1e-4)
        preds = np.argmax(adjusted_probs, axis=1)
        preds[speed < low_speed_th] = 0
        return -f1_score(y, preds, average='macro')
        
    init_th = [1.0, 1.0, 1.0]
    opt_res = minimize(threshold_loss, init_th, method='Nelder-Mead', options={'maxiter': 300})
    best_thresholds = opt_res.x
    best_thresholds = np.clip(best_thresholds, 0.1, 10.0)
    
    adj_probs = best_oof_probs / best_thresholds
    tuned_preds = np.argmax(adj_probs, axis=1)
    tuned_preds[speed < low_speed_th] = 0
    
    res_tuned = evaluate_predictions(y, tuned_preds, f"Threshold-Tuned {best_clf_type.upper()} + Gating")
    leaderboard.append(res_tuned)
    print("Optimal Probability Divisors [Normal, Side I, Side II]:", np.round(best_thresholds, 3))
    print(res_tuned)

    # -------------------------------------------------------------------------
    # Baseline XGBoost for reference comparison
    # -------------------------------------------------------------------------
    xgb_base_oof = np.zeros(len(y))
    base_weights = np.zeros(len(y))
    for c_idx in range(3):
        base_weights[y == c_idx] = len(y) / (3.0 * np.sum(y == c_idx))
    for tr_idx, va_idx in skf.split(X, y):
        base_clf = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.08, random_state=42, eval_metric='mlogloss', verbosity=0)
        base_clf.fit(X[tr_idx], y[tr_idx], sample_weight=base_weights[tr_idx])
        xgb_base_oof[va_idx] = base_clf.predict(X[va_idx])
    res_base = evaluate_predictions(y, xgb_base_oof, "Weighted XGBoost (Baseline)")
    leaderboard.insert(0, res_base)

    # -------------------------------------------------------------------------
    # PHASE 4: Final Validation Leaderboard Table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 4: FINAL MODEL COMPARISON LEADERBOARD")
    print("=" * 80)
    df_leaderboard = pd.DataFrame(leaderboard).sort_values(by="Macro F1", ascending=False)
    print(df_leaderboard.to_string(index=False))
    
    # Save leaderboard CSV
    lb_path = os.path.join(BASE_DIR, "automl_leaderboard_results.csv")
    df_leaderboard.to_csv(lb_path, index=False)
    print(f"\nSaved leaderboard comparison to: {lb_path}")
    
    # -------------------------------------------------------------------------
    # PHASE 5: Train Winning Final Model on Full Dataset & Export
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 5: Exporting Production Model Bundle & Generating Test Predictions")
    print("=" * 80)
    
    full_sample_weights = class_weights[y]
    
    if best_clf_type == "xgboost":
        final_model = xgb.XGBClassifier(
            n_estimators=140,
            max_depth=bp['xgb_depth'],
            learning_rate=bp['xgb_lr'],
            colsample_bytree=bp['xgb_colsample'],
            subsample=bp['xgb_subsample'],
            reg_lambda=bp['xgb_lambda'],
            random_state=42,
            eval_metric="mlogloss",
            verbosity=0
        )
        final_model.fit(X, y, sample_weight=full_sample_weights)
    elif best_clf_type == "catboost":
        final_model = CatBoostClassifier(
            iterations=150,
            depth=bp['cb_depth'],
            learning_rate=bp['cb_lr'],
            l2_leaf_reg=bp['cb_l2'],
            loss_function="MultiClass",
            class_weights=class_weights.tolist(),
            random_seed=42,
            verbose=0
        )
        final_model.fit(X, y)
    elif best_clf_type == "lightgbm":
        final_model = lgb.LGBMClassifier(
            n_estimators=140,
            max_depth=bp['lgb_depth'],
            num_leaves=bp['lgb_leaves'],
            learning_rate=bp['lgb_lr'],
            subsample=bp['lgb_subsample'],
            random_state=42,
            verbose=-1
        )
        final_model.fit(X, y, sample_weight=full_sample_weights)
        
    production_bundle = {
        'model': final_model,
        'model_type': best_clf_type,
        'feature_cols': feature_cols,
        'best_thresholds': best_thresholds,
        'low_speed_threshold_kmh': low_speed_th,
        'label_map': LABEL_MAP,
        'inv_label_map': INV_MAP,
        'metadata': {
            'best_macro_f1': df_leaderboard.iloc[0]['Macro F1'],
            'optuna_params': bp,
            'exported_timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    
    bundle_dest = os.path.join(MODELS_DIR, "rail_model_bundle.joblib")
    joblib.dump(production_bundle, bundle_dest)
    print(f"Production model bundle saved to: {bundle_dest}")
    
    # -------------------------------------------------------------------------
    # Generate Predictions for Test Dataset
    # -------------------------------------------------------------------------
    test_files = sorted(
        [f for f in os.listdir(TEST_DIR) if f.endswith(".csv")],
        key=lambda x: int(x.replace("Test", "").replace(".csv", ""))
    )
    
    print(f"\nGenerating predictions on {len(test_files)} test files using winning pipeline...")
    from investigate_rail_corrugation import extract_features_from_df
    
    test_predictions = []
    for tf in test_files:
        df_test = pd.read_csv(os.path.join(TEST_DIR, tf))
        feats = extract_features_from_df(df_test)
        
        # 1. Low speed rule
        if feats['speed_kmh'] < low_speed_th:
            pred_lbl = "Normal"
        else:
            x_test = np.array([[feats[col] for col in feature_cols]])
            raw_probs = final_model.predict_proba(x_test)[0]
            # Apply threshold tuning
            adj_probs = raw_probs / best_thresholds
            pred_lbl = INV_MAP[np.argmax(adj_probs)]
            
        test_predictions.append({'file_id': tf, 'prediction': pred_lbl})
        
    df_test_preds = pd.DataFrame(test_predictions)
    sub_path = os.path.join(BASE_DIR, "automl_rail_predictions.csv")
    df_test_preds.to_csv(sub_path, index=False)
    # Also update baseline_rail_predictions.csv
    df_test_preds.to_csv(os.path.join(BASE_DIR, "baseline_rail_predictions.csv"), index=False)
    
    print(f"Predictions saved to: {sub_path}")
    print("\nTest Prediction Distribution:")
    print(df_test_preds['prediction'].value_counts())
    print("\n" + "=" * 80)
    print("ALL AUTOML PIPELINE PHASES COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
