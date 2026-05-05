import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm import tqdm
import joblib
import optuna

# Mapping environment for modular execution
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import TEST_CUTOFF_DT
from src.tools import compute_cluster_metrics, wmape, plot_cluster_portfolio, analyze_time_periods, load_processed_data

# Suppress statsmodels warnings for clean terminal output
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")


def preprocess_and_split(df_long):
    print("Preparing train/test split and log-scaling for SARIMAX...")
    
    train_raw = df_long[df_long['Week'] < TEST_CUTOFF_DT].copy()
    test_raw  = df_long[df_long['Week'] >= TEST_CUTOFF_DT].copy()
    
    # 1. Log Scaling for Quantity (Clip at 0 to prevent negative values)
    print("Applying log1p scaling to Quantity...")
    train_raw['Quantity_Scaled'] = np.log1p(train_raw['Quantity'].clip(lower=0))
    test_raw['Quantity_Scaled'] = np.log1p(test_raw['Quantity'].clip(lower=0))

    # 2. Define External Regressors
    # We include regressors just like in Prophet. SARIMAX will handle the time series autocorrelation internally.
    regressors = ['is_christmas_window', 'is_on_promotion', 'price_percent_change', 'lag_1', 'lag_4']
    
    # Fill NAs for regressors before aggregation
    train_raw[regressors] = train_raw[regressors].fillna(0)
    test_raw[regressors] = test_raw[regressors].fillna(0)

    # 3. Aggregation by Cluster
    print("Aggregating data and regressors by Cluster for SARIMAX training...")
    if 'profile_cluster_id' not in train_raw.columns:
        raise ValueError("Error: profile_cluster_id not found in data!")

    cols_to_agg = ['Quantity_Scaled'] + regressors
    train_agg = train_raw.groupby(['profile_cluster_id', 'Week'], observed=True)[cols_to_agg].mean().reset_index()
    test_agg  = test_raw.groupby(['profile_cluster_id', 'Week'], observed=True)[cols_to_agg].mean().reset_index()
    
    # Sort chronologically, critical for ARIMA models
    train_agg = train_agg.sort_values(by=['profile_cluster_id', 'Week'])
    test_agg  = test_agg.sort_values(by=['profile_cluster_id', 'Week'])
    
    return train_agg, test_agg, test_raw, regressors


def tune_hyperparameters(train_agg, regressors, n_trials=50):
    """
    Uses Optuna to find the best SARIMAX parameters (p, d, q) and (P, D, Q).
    We restrict the search space and trials because SARIMAX fitting is computationally expensive.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Time-aware split: last 12 weeks as validation
    val_cutoff = train_agg['Week'].max() - pd.Timedelta(weeks=12)
    train_part = train_agg[train_agg['Week'] < val_cutoff]
    val_part = train_agg[train_agg['Week'] >= val_cutoff]
    
    unique_clusters = sorted(train_part['profile_cluster_id'].dropna().unique())
    
    def objective(trial):
        # Base terms
        p = trial.suggest_int("p", 0, 3)
        d = trial.suggest_int("d", 0, 2)
        q = trial.suggest_int("q", 0, 3)
        
        # Seasonal terms (s=52 for weekly)
        P = trial.suggest_int("P", 0, 2)
        D = trial.suggest_int("D", 0, 1)
        Q = trial.suggest_int("Q", 0, 2)
        
        all_actual = []
        all_predicted = []
        
        for cluster_id in unique_clusters:
            df_c = train_part[train_part['profile_cluster_id'] == cluster_id]
            val_c = val_part[val_part['profile_cluster_id'] == cluster_id]
            
            if len(df_c) < 52 or len(val_c) == 0:
                continue # Skip if not enough history for a full 52-week seasonal cycle
            
            y_train = df_c['Quantity_Scaled'].values
            X_train = df_c[regressors].values
            
            y_val = val_c['Quantity_Scaled'].values
            X_val = val_c[regressors].values
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = SARIMAX(
                        y_train,
                        exog=X_train,
                        order=(p, d, q),
                        seasonal_order=(P, D, Q, 52),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    fit_res = model.fit(disp=False, maxiter=50)
                
                fcst_scaled = fit_res.forecast(steps=len(y_val), exog=X_val)
                fcst_scaled = np.clip(fcst_scaled, a_min=None, a_max=20.0)
                
            except Exception:
                # Fallback to naive mean if SARIMAX fails (common with sparsity)
                fcst_scaled = np.full(len(y_val), np.mean(y_train))
                
            pred_qty = np.maximum(np.expm1(fcst_scaled), 0)
            actual_qty = np.maximum(np.expm1(y_val), 0)
            
            all_actual.extend(actual_qty)
            all_predicted.extend(pred_qty)
        
        if len(all_actual) == 0:
            return 999.0
            
        return wmape(np.array(all_actual), np.array(all_predicted))
    
    print(f"\nOptuna: Tuning SARIMAX ({n_trials} trials)...")
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best = study.best_params
    print(f"Best SARIMAX params (WMAPE={study.best_value:.2f}%): {best}")
    
    return best


def train_models(train_agg, regressors, params=None):
    print(f"Training SARIMAX models for {train_agg['profile_cluster_id'].nunique()} clusters...")
    cluster_models = {}
    unique_clusters = sorted(train_agg['profile_cluster_id'].dropna().unique())

    # Defaults if no params provided
    if params is None:
        params = {"p": 0, "d": 2, "q": 1, "P": 1, "D": 0, "Q": 1}

    order = (params["p"], params["d"], params["q"])
    seasonal_order = (params["P"], params["D"], params["Q"], 52)

    for cluster_id in tqdm(unique_clusters, desc="Training"):
        df_c = train_agg[train_agg['profile_cluster_id'] == cluster_id]
        
        y_train = df_c['Quantity_Scaled'].values
        X_train = df_c[regressors].values
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(
                    y_train,
                    exog=X_train,
                    order=order,
                    seasonal_order=seasonal_order if len(df_c) >= 70 else (0,0,0,0),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fit_res = model.fit(disp=False, maxiter=100)
            cluster_models[cluster_id] = fit_res
        except Exception as e:
            print(f"Failed to fit cluster {cluster_id}: {e}")
            
    return cluster_models


def predict_models(cluster_models, test_agg, test_raw, regressors):
    print("Generating forecasts and un-scaling to raw Quantity...")
    
    all_cluster_forecasts = []
    
    for cluster_id, fit_res in cluster_models.items():
        df_test_c = test_agg[test_agg['profile_cluster_id'] == cluster_id]
        if len(df_test_c) == 0:
            continue
            
        X_test = df_test_c[regressors].values
        
        fcst_scaled = fit_res.forecast(steps=len(df_test_c), exog=X_test)
            
        fcst_df = pd.DataFrame({
            'profile_cluster_id': cluster_id,
            'Week': df_test_c['Week'].values,
            'Predicted_Quantity_Scaled': fcst_scaled
        })
        all_cluster_forecasts.append(fcst_df)

    if len(all_cluster_forecasts) == 0:
        return test_raw
        
    global_forecasts = pd.concat(all_cluster_forecasts, ignore_index=True)

    # Merge to Individual SKUs
    test_raw = test_raw.drop(columns=['Predicted_Quantity_Scaled'], errors='ignore')
    test_raw = test_raw.merge(global_forecasts, on=['profile_cluster_id', 'Week'], how='left')
    
    # Inverse Scaling (expm1)
    print("Inverse transforming predictions (expm1)...")
    pred_scaled = test_raw['Predicted_Quantity_Scaled'].fillna(0).values
    pred_scaled = np.clip(pred_scaled, a_min=None, a_max=20.0)
    unscaled = np.expm1(pred_scaled)
    
    test_raw['Predicted_Qty'] = np.maximum(unscaled, 0)
    test_raw['Actual_Qty'] = test_raw['Quantity']
                
    return test_raw


def evaluate_models(test_raw):
    print("\nEvaluating Portfolio Performance...")
    
    test_raw['Cluster'] = test_raw['profile_cluster_id']
    test_raw['Date'] = test_raw['Week']
    
    portfolio_eval = test_raw.dropna(subset=['Actual_Qty', 'Predicted_Qty'])[['Cluster', 'StockCode', 'Date', 'Actual_Qty', 'Predicted_Qty']].copy()
    summary = compute_cluster_metrics(portfolio_eval)

    return portfolio_eval, summary


def save_artifacts(cluster_models, regressors, sku_clusters, best_params=None, artifacts_dir="../agent/artifacts"):
    print(f"Saving Cluster SARIMAX artifacts to {artifacts_dir}...")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    file_name = "sarimax_cluster_models.pkl"
    path = os.path.join(artifacts_dir, file_name)
    
    artifact = {
        "models": cluster_models,
        "features": list(regressors),
        "scaler": None,
        "metadata": {
            "sku_clusters": {k: v for k, v in sku_clusters.items()},
            "best_params": best_params,
            "model_type": "sarimax"
        }
    }
    
    joblib.dump(artifact, path)
    print(f"Successfully saved {path}")

    if best_params:
        json_path = os.path.join(artifacts_dir, "sarimax_best_params.json")
        with open(json_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"Saved best params to {json_path}")


def run_sarimax_pipeline(file_path, plot=False, tune=False):
    df_long = load_processed_data(file_path)
    
    train_agg, test_agg, test_raw, regressors = preprocess_and_split(df_long)
    
    best_params = None
    if tune:
        best_params = tune_hyperparameters(train_agg, regressors, n_trials=50)
    
    cluster_models = train_models(train_agg, regressors, params=best_params)
    
    test_raw = predict_models(cluster_models, test_agg, test_raw, regressors)
    cluster_eval, summary = evaluate_models(test_raw)
    
    sku_clusters = df_long.drop_duplicates(subset=['StockCode']).set_index('StockCode')['profile_cluster_id'].to_dict()
    artifacts_dir = os.path.join(PROJECT_ROOT, 'agent', 'artifacts')
    save_artifacts(cluster_models, regressors, sku_clusters, best_params=best_params, artifacts_dir=artifacts_dir)
    
    # Save per-SKU metrics for the model selector
    sku_wmape = {}
    for sku, group in cluster_eval.groupby('StockCode'):
        act = group['Actual_Qty'].values
        prd = group['Predicted_Qty'].values
        if act.sum() > 0:
            sku_wmape[sku] = float(wmape(act, prd))
    with open(os.path.join(artifacts_dir, "sarimax_sku_wmape.json"), "w") as f:
        json.dump(sku_wmape, f, indent=2)
    
    if plot:
        plot_cluster_portfolio(cluster_eval, summary, model_label="SARIMAX Forecast")
        analyze_time_periods(test_raw)
        
    return cluster_models, test_raw, cluster_eval, summary


if __name__ == "__main__":
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed_retail_data.parquet")
    # Default to 10 trials for SARIMAX to keep execution time reasonable
    _, _, _, summary = run_sarimax_pipeline(DATA_PATH, plot=False, tune=True)
    print("\n=== SARIMAX Evaluation Summary ===")
    print(summary.to_markdown())
