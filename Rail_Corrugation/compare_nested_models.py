import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

warnings.filterwarnings('ignore')

CSV_PATH = r'c:\Users\Rald999\Documents\GitHub\nebulax_p3\Rail_Corrugation\rail_all_features.csv'
df = pd.read_csv(CSV_PATH)
feature_cols = [c for c in df.columns if c not in ['filename', 'label']]
X = df[feature_cols].values
LABEL_MAP = {'Normal': 0, 'Side I': 1, 'Side II': 2}
y = df['label'].map(LABEL_MAP).values
speed = df['speed_kmh'].values

SEEDS = [42, 101, 2024]
models = ['LightGBM', 'LightGBMXT', 'CatBoost', 'XGBoost', 'Ensemble_LGB_XT', 'Ensemble_TriBlend']

def optimize_divisors(probs, y_true, spd):
    best_f1 = -1.0
    best_divs = np.array([1.0, 1.0, 1.0])
    for s1_d in np.linspace(0.12, 0.45, 25):
        for s2_d in np.linspace(0.60, 1.20, 20):
            divs = np.array([1.0, s1_d, s2_d])
            adj = probs / divs
            pr = np.argmax(adj, axis=1)
            pr[spd < 10.0] = 0
            sc = f1_score(y_true, pr, average='macro')
            if sc > best_f1:
                best_f1 = sc
                best_divs = divs
    return best_divs

results = {m: {'macro_f1': [], 'side1_f1': [], 'side2_f1': [], 'norm_f1': [], 'acc': []} for m in models}

for seed in SEEDS:
    outer_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    outer_preds = {m: np.zeros(len(y), dtype=int) for m in models}
    
    for tr_out_idx, val_out_idx in outer_skf.split(X, y):
        X_tr, y_tr = X[tr_out_idx], y[tr_out_idx]
        speed_tr = speed[tr_out_idx]
        X_val, y_val = X[val_out_idx], y[val_out_idx]
        speed_val = speed[val_out_idx]
        
        inner_skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
        inner_oof = {m: np.zeros((len(y_tr), 3)) for m in models}
        
        for tr_in_idx, val_in_idx in inner_skf.split(X_tr, y_tr):
            X_tri, y_tri = X_tr[tr_in_idx], y_tr[tr_in_idx]
            X_vai, y_vai = X_tr[val_in_idx], y_tr[val_in_idx]
            
            c_counts = np.bincount(y_tri, minlength=3)
            weights = len(y_tri) / (3.0 * np.maximum(c_counts, 1))
            sw = weights[y_tri]
            
            lgb_m = lgb.LGBMClassifier(n_estimators=150, num_leaves=18, max_depth=5, learning_rate=0.06, subsample=0.85, colsample_bytree=0.85, random_state=seed, verbose=-1).fit(X_tri, y_tri, sample_weight=sw)
            xt_m = lgb.LGBMClassifier(n_estimators=150, num_leaves=24, max_depth=6, learning_rate=0.05, subsample=0.80, colsample_bytree=0.75, extra_trees=True, random_state=seed+10, verbose=-1).fit(X_tri, y_tri, sample_weight=sw)
            cb_m = CatBoostClassifier(iterations=150, depth=6, learning_rate=0.08, l2_leaf_reg=3.0, loss_function='MultiClass', class_weights=weights.tolist(), random_seed=seed, verbose=0).fit(X_tri, y_tri)
            xgb_m = xgb.XGBClassifier(n_estimators=140, max_depth=4, learning_rate=0.07, colsample_bytree=0.85, subsample=0.85, reg_lambda=2.0, random_state=seed, eval_metric='mlogloss', verbosity=0).fit(X_tri, y_tri, sample_weight=sw)
            
            p_lgb = lgb_m.predict_proba(X_vai)
            p_xt = xt_m.predict_proba(X_vai)
            p_cb = cb_m.predict_proba(X_vai)
            p_xgb = xgb_m.predict_proba(X_vai)
            
            inner_oof['LightGBM'][val_in_idx] = p_lgb
            inner_oof['LightGBMXT'][val_in_idx] = p_xt
            inner_oof['CatBoost'][val_in_idx] = p_cb
            inner_oof['XGBoost'][val_in_idx] = p_xgb
            inner_oof['Ensemble_LGB_XT'][val_in_idx] = 0.6 * p_lgb + 0.4 * p_xt
            inner_oof['Ensemble_TriBlend'][val_in_idx] = 0.45 * p_lgb + 0.35 * p_cb + 0.20 * p_xgb
            
        best_divs = {m: optimize_divisors(inner_oof[m], y_tr, speed_tr) for m in models}
        
        c_counts_out = np.bincount(y_tr, minlength=3)
        weights_out = len(y_tr) / (3.0 * np.maximum(c_counts_out, 1))
        sw_out = weights_out[y_tr]
        
        out_lgb = lgb.LGBMClassifier(n_estimators=150, num_leaves=18, max_depth=5, learning_rate=0.06, subsample=0.85, colsample_bytree=0.85, random_state=seed, verbose=-1).fit(X_tr, y_tr, sample_weight=sw_out)
        out_xt = lgb.LGBMClassifier(n_estimators=150, num_leaves=24, max_depth=6, learning_rate=0.05, subsample=0.80, colsample_bytree=0.75, extra_trees=True, random_state=seed+10, verbose=-1).fit(X_tr, y_tr, sample_weight=sw_out)
        out_cb = CatBoostClassifier(iterations=150, depth=6, learning_rate=0.08, l2_leaf_reg=3.0, loss_function='MultiClass', class_weights=weights_out.tolist(), random_seed=seed, verbose=0).fit(X_tr, y_tr)
        out_xgb = xgb.XGBClassifier(n_estimators=140, max_depth=4, learning_rate=0.07, colsample_bytree=0.85, subsample=0.85, reg_lambda=2.0, random_state=seed, eval_metric='mlogloss', verbosity=0).fit(X_tr, y_tr, sample_weight=sw_out)
        
        pv_lgb = out_lgb.predict_proba(X_val)
        pv_xt = out_xt.predict_proba(X_val)
        pv_cb = out_cb.predict_proba(X_val)
        pv_xgb = out_xgb.predict_proba(X_val)
        
        p_val_dict = {
            'LightGBM': pv_lgb,
            'LightGBMXT': pv_xt,
            'CatBoost': pv_cb,
            'XGBoost': pv_xgb,
            'Ensemble_LGB_XT': 0.6 * pv_lgb + 0.4 * pv_xt,
            'Ensemble_TriBlend': 0.45 * pv_lgb + 0.35 * pv_cb + 0.20 * pv_xgb
        }
        
        for m in models:
            adj_p = p_val_dict[m] / best_divs[m]
            pr = np.argmax(adj_p, axis=1)
            pr[speed_val < 10.0] = 0
            outer_preds[m][val_out_idx] = pr
            
    for m in models:
        f1_m = f1_score(y, outer_preds[m], average='macro')
        acc_m = accuracy_score(y, outer_preds[m])
        pc = f1_score(y, outer_preds[m], average=None)
        results[m]['macro_f1'].append(f1_m)
        results[m]['norm_f1'].append(pc[0])
        results[m]['side1_f1'].append(pc[1])
        results[m]['side2_f1'].append(pc[2])
        results[m]['acc'].append(acc_m)

print('\n' + '='*85)
print('STRICT NESTED CROSS-VALIDATION BENCHMARK COMPARISON')
print('='*85)
header = "{:<20} | {:<18} | {:<10} | {:<11} | {:<10}".format("Model", "Nested Macro F1", "Side I F1", "Side II F1", "Accuracy")
print(header)
print('-'*85)
for m in models:
    mf1_mean = np.mean(results[m]['macro_f1'])
    mf1_std = np.std(results[m]['macro_f1'])
    s1_mean = np.mean(results[m]['side1_f1'])
    s2_mean = np.mean(results[m]['side2_f1'])
    acc_mean = np.mean(results[m]['acc']) * 100
    row = "{:<20} | {:.4f} +/- {:.4f}   | {:.4f}     | {:.4f}      | {:.2f}%".format(
        m, mf1_mean, mf1_std, s1_mean, s2_mean, acc_mean
    )
    print(row)
