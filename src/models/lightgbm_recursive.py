import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import joblib
import logging
import lightgbm as lgb
import optuna

# Ensure project root is in sys.path for absolute imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import TEST_CUTOFF_DT
from src.tools import compute_cluster_metrics, wmape, plot_cluster_portfolio, analyze_time_periods, load_processed_data


def preprocess_and_split(df_long):
    print("Feature Engineering and Train/Test Split...")
    
    # LightGBM handles categorical features natively, but they must be explicitly typed as 'category'
    cat_cols = ['volume_tier', 'semantic_cluster_name', 'demand_class']
    for col in cat_cols:
        if col in df_long.columns:
            df_long[col] = df_long[col].astype('category')
            
    # We do NOT use pd.get_dummies() here because LightGBM's native categorical handling is faster and more accurate
    
    # We rely on the global test_cutoff defined in src/config.py
    train = df_long[df_long['Week'] < TEST_CUTOFF_DT].copy()
    test  = df_long[df_long['Week'] >= TEST_CUTOFF_DT].copy()
    
    train = train.sort_values(by=['StockCode', 'Week'])
    test  = test.sort_values(by=['StockCode', 'Week'])

    # Features to scale per SKU
    scale_cols = [f'lag_{l}' for l in (1, 4, 13, 52)] + \
                 [f'rmean_{w}' for w in (4, 13)] + [f'rstd_{w}' for w in (4, 13)] + \
                 ['return_rate_4w', 'return_rate_13w'] + \
                 ['price_weekly', 'price_percent_change'] + \
                 ['ADI', 'CV2', 'share_zero_weeks']
                 
    # Ensure they exist
    scale_cols = [c for c in scale_cols if c in train.columns]

    # Log transform the target natively
    train['Quantity_Scaled'] = np.log1p(train['Quantity'].clip(lower=0))
    test['Quantity_Scaled']  = np.log1p(test['Quantity'].clip(lower=0))

    print("Applying Global Scaling (Log1p for volumes, MinMax for ratios)...")
    
    # 1. Log-transform volume-based features
    vol_features = [f'lag_{l}' for l in (1, 4, 13, 52)] + [f'rmean_{w}' for w in (4, 13)] + [f'rstd_{w}' for w in (4, 13)]
    for col in vol_features:
        if col in train.columns:
            train[f'{col}_Scaled'] = np.log1p(train[col].clip(lower=0).fillna(0))
            test[f'{col}_Scaled']  = np.log1p(test[col].clip(lower=0).fillna(0))

    # 2. Global MinMax for ratios
    ratio_features = ['return_rate_4w', 'return_rate_13w', 'price_weekly', 'price_percent_change', 'ADI', 'CV2', 'share_zero_weeks']
    ratio_features = [c for c in ratio_features if c in train.columns]
    
    global_scaler = MinMaxScaler()
    
    train[ratio_features] = train[ratio_features].fillna(0)
    test[ratio_features]  = test[ratio_features].fillna(0)
    
    train_scaled = global_scaler.fit_transform(train[ratio_features])
    test_scaled  = global_scaler.transform(test[ratio_features])
    
    for i, col in enumerate(ratio_features):
        train[f'{col}_Scaled'] = train_scaled[:, i]
        test[f'{col}_Scaled']  = test_scaled[:, i]

    train = train.dropna(subset=['Quantity_Scaled'])

    # Columns to drop for X (Same leakage prevention as Linear Regression)
    cols_to_drop = [
        'Week', 'StockCode', 'Quantity', 'Quantity_Scaled', 'qty_returned', 'Revenue',
        'week_of_year', 'month', 'quarter', 'year',
        'price_median', 'mean_basket_size', 'n_unique_customers', 'country_uk_share'
    ] + scale_cols + ['desc_canonical', 'embedding']

    cols_to_drop_train = [c for c in cols_to_drop if c in train.columns]

    X_train = train.drop(columns=cols_to_drop_train)
    y_train = train['Quantity_Scaled']

    test = test.sort_values(by=['StockCode', 'Week'])
    X_test = test.drop(columns=cols_to_drop_train)

    print(f"Training shape: {X_train.shape}")
    print(f"Testing shape:  {X_test.shape}")

    # We exclude profile_cluster_id from feature_cols because we segment on it
    feature_cols = X_train.drop(columns=['profile_cluster_id'], errors='ignore').columns.tolist()

    return train, test, X_train, y_train, X_test, feature_cols, global_scaler


def tune_hyperparameters(X_train, y_train, train, n_trials=50):
    """
    Uses Optuna to find the best LightGBM hyperparameters.
    
    Strategy: Time-aware validation split — the last 12 weeks of training data
    are held out as validation. We train on the rest and evaluate WMAPE.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Time-aware split: last 12 weeks as validation
    val_cutoff = train['Week'].max() - pd.Timedelta(weeks=12)
    train_mask = train['Week'] < val_cutoff
    val_mask = train['Week'] >= val_cutoff
    
    X_tr = X_train[train_mask].drop(columns=['profile_cluster_id'], errors='ignore')
    y_tr = y_train[train_mask]
    X_val = X_train[val_mask].drop(columns=['profile_cluster_id'], errors='ignore')
    y_val_scaled = y_train[val_mask]
    y_val_qty = train.loc[val_mask, 'Quantity'].values
    
    def objective(trial):
        params = {
            "objective": "tweedie",
            "tweedie_variance_power": trial.suggest_float("tweedie_variance_power", 1.0, 1.8),
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "random_state": 42,
            "verbosity": -1,
            "n_jobs": -1,
        }
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr)
        
        preds_scaled = model.predict(X_val)
        preds_scaled = np.clip(preds_scaled, a_min=None, a_max=20.0)
        preds_qty = np.maximum(np.expm1(preds_scaled), 0)
        
        return wmape(y_val_qty, preds_qty)
    
    print(f"\nOptuna: Tuning LightGBM ({n_trials} trials)...")
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best = study.best_params
    print(f"Best LightGBM params (WMAPE={study.best_value:.2f}%): {best}")
    
    return best


def train_models(X_train, y_train, train, params=None):
    print("Training LightGBM models per Seasonal Profile Cluster...")
    cluster_models = {}
        
    unique_clusters = train['profile_cluster_id'].dropna().unique()

    # Merge tuned params with essential defaults
    model_params = {
        "objective": "tweedie",
        "tweedie_variance_power": 1.2,
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
    }
    if params:
        model_params.update(params)

    for cluster_id in sorted(unique_clusters):
        cluster_mask = train['profile_cluster_id'] == cluster_id
        X_train_cluster = X_train[cluster_mask].drop(columns=['profile_cluster_id'], errors='ignore')
        y_train_cluster = y_train[cluster_mask]

        if len(X_train_cluster) == 0:
            continue

        model = lgb.LGBMRegressor(**model_params)
        
        # LightGBM handles NaNs automatically, no need for np.nan_to_num!
        model.fit(X_train_cluster, y_train_cluster)
        cluster_models[cluster_id] = model
        print(f" - Model for Cluster {int(cluster_id)} trained on {len(X_train_cluster)} historical rows.")

    return cluster_models



def predict_models(cluster_models, test, X_test):
    print("Predicting on Test Set...")
    test['Predicted_Quantity_Scaled'] = np.nan

    for cluster_id, model in cluster_models.items():
        cluster_mask = test['profile_cluster_id'] == cluster_id
        X_test_cluster = X_test[cluster_mask].drop(columns=['profile_cluster_id'], errors='ignore')
        
        if len(X_test_cluster) > 0:
            preds = model.predict(X_test_cluster)
            test.loc[cluster_mask, 'Predicted_Quantity_Scaled'] = preds

    print("Applying physical constraints (Capping at 0)...")
    test['Predicted_Quantity_Scaled'] = np.maximum(test['Predicted_Quantity_Scaled'].fillna(0), 0)

    print("Predictions Complete!")
    return test


def evaluate_models(test):
    print("\nEvaluating model (raw Quantity)...")
    
    # Vectorized inverse scaling instead of slow SKU looping
    valid = test['Quantity'].notna() & test['Predicted_Quantity_Scaled'].notna()
    
    y_true_qty = test.loc[valid, 'Quantity'].values
    y_pred_scaled = test.loc[valid, 'Predicted_Quantity_Scaled'].values
    
    # Prevent expm1 overflow
    y_pred_scaled = np.clip(y_pred_scaled, a_min=None, a_max=20.0) 
    y_pred_qty = np.maximum(np.expm1(y_pred_scaled), 0)
    
    test.loc[valid, 'Actual_Qty'] = y_true_qty
    test.loc[valid, 'Predicted_Qty'] = y_pred_qty

    # Pass the raw, item-level predictions to calculate Median MAPE and WMAPE
    test['Cluster'] = test['profile_cluster_id']
    test['Date'] = test['Week']
    
    cluster_eval = test.dropna(subset=['Actual_Qty', 'Predicted_Qty'])[['Cluster', 'StockCode', 'Date', 'Actual_Qty', 'Predicted_Qty']].copy()

    summary = compute_cluster_metrics(cluster_eval)

    return cluster_eval, summary


def save_artifacts(cluster_models, feature_cols, sku_clusters, global_scaler=None, best_params=None, artifacts_dir="../agent/artifacts"):
    print(f"Saving Cluster LightGBM artifacts to {artifacts_dir}...")
    os.makedirs(artifacts_dir, exist_ok=True)

    file_name = "lgb_cluster_models.pkl"
    path = os.path.join(artifacts_dir, file_name)

    artifact = {
        "cluster_models": cluster_models,
        "feature_cols": list(feature_cols),
        "sku_clusters": {k: v for k, v in sku_clusters.items()},
        "global_scaler": global_scaler,
        "best_params": best_params,
    }
    
    joblib.dump(artifact, path)
    print(f"Successfully saved {path}")

    # Also save best params as readable JSON
    if best_params:
        json_path = os.path.join(artifacts_dir, "lgb_best_params.json")
        with open(json_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"Saved best params to {json_path}")


def run_lgb_pipeline(file_path, plot=False, tune=False):
    """
    Complete pipeline to load data, (optionally tune), train models, predict, evaluate, and visualize results.
    """
    df_long = load_processed_data(file_path)
    train, test, X_train, y_train, X_test, feature_cols, global_scaler = preprocess_and_split(df_long)

    best_params = None
    if tune:
        best_params = tune_hyperparameters(X_train, y_train, train)

    cluster_models = train_models(X_train, y_train, train, params=best_params)
    test = predict_models(cluster_models, test, X_test)
    cluster_eval, summary = evaluate_models(test)

    sku_clusters = df_long.drop_duplicates(subset=['StockCode']).set_index('StockCode')['profile_cluster_id'].to_dict()
    artifacts_dir = os.path.join(PROJECT_ROOT, 'agent', 'artifacts')
    save_artifacts(cluster_models, feature_cols, sku_clusters, global_scaler=global_scaler, best_params=best_params, artifacts_dir=artifacts_dir)
    
    # Save per-SKU metrics for the model selector
    sku_wmape = {}
    for sku, group in cluster_eval.groupby('StockCode'):
        act = group['Actual_Qty'].values
        prd = group['Predicted_Qty'].values
        if act.sum() > 0:
            sku_wmape[sku] = float(wmape(act, prd))
    with open(os.path.join(artifacts_dir, "lgb_sku_wmape.json"), "w") as f:
        json.dump(sku_wmape, f, indent=2)
    
    if plot:
        plot_cluster_portfolio(cluster_eval, summary, model_label="LightGBM Forecast")
        analyze_time_periods(test)
    
    return cluster_models, test, cluster_eval, summary

if __name__ == "__main__":
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed_retail_data.parquet")
    # By default, we run with tuning enabled when called directly from terminal
    _, _, _, summary = run_lgb_pipeline(DATA_PATH, plot=False, tune=True)
    print("\n=== LightGBM Evaluation Summary (Optuna Tuned) ===")
    print(summary.to_markdown())