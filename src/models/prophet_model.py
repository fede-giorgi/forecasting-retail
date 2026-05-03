"""
prophet_model.py
----------------
Modularized implementation of the retail forecasting model using Facebook Prophet.
Trains one model per seasonal profile cluster (averaged shape) and un-scales for individual SKUs.

Features:
- UK holidays and seasonality.
- Log1p scaling to handle intermittent demand and extreme outliers.
- Performance evaluation at the portfolio level.
"""

import os
import sys
import numpy as np
import pandas as pd
from prophet import Prophet
from tqdm import tqdm
import joblib
import logging

# Mapping environment for modular execution
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import TEST_CUTOFF_DT
from src.tools import load_processed_data, compute_cluster_metrics, plot_cluster_portfolio, analyze_time_periods

# Suppress Prophet/cmdstanpy verbose logging
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)


def preprocess_and_split(df_long):
    print("Preparing train/test split and log-scaling...")
    
    train_raw = df_long[df_long['Week'] < TEST_CUTOFF_DT].copy()
    test_raw  = df_long[df_long['Week'] >= TEST_CUTOFF_DT].copy()
    
    # 1. Log Scaling for Quantity (Clip at 0 to prevent negative values)
    print("Applying log1p scaling to Quantity...")
    train_raw['Quantity_Scaled'] = np.log1p(train_raw['Quantity'].clip(lower=0))
    test_raw['Quantity_Scaled'] = np.log1p(test_raw['Quantity'].clip(lower=0))

    # 2. Define External Regressors to include in Prophet
    # We select key exogenous features. Since Prophet models time inherently via Fourier terms, 
    # we avoid redundant temporal features like 'month' or 'sin_woy'.
    regressors = ['is_christmas_window', 'is_on_promotion', 'price_percent_change', 'lag_1', 'lag_4']
    
    # Fill NAs for regressors before aggregation
    train_raw[regressors] = train_raw[regressors].fillna(0)
    test_raw[regressors] = test_raw[regressors].fillna(0)

    # 3. Aggregation by Cluster for Prophet
    print("Aggregating data and regressors by Cluster for Prophet training...")
    if 'profile_cluster_id' not in train_raw.columns:
        raise ValueError("Error: profile_cluster_id not found in data!")

    # We aggregate both the target and the external regressors by taking the mean across the cluster
    cols_to_agg = ['Quantity_Scaled'] + regressors
    train_agg = train_raw.groupby(['profile_cluster_id', 'Week'], observed=True)[cols_to_agg].mean().reset_index()
    test_agg  = test_raw.groupby(['profile_cluster_id', 'Week'], observed=True)[cols_to_agg].mean().reset_index()
    
    train_agg = train_agg.rename(columns={'Week': 'ds', 'Quantity_Scaled': 'y', 'profile_cluster_id': 'Cluster'})
    test_agg  = test_agg.rename(columns={'Week': 'ds', 'Quantity_Scaled': 'y', 'profile_cluster_id': 'Cluster'})
    
    return train_agg, test_agg, test_raw, regressors


def train_models(train_agg, regressors):
    print(f"Training Prophet models for {train_agg['Cluster'].nunique()} clusters...")
    cluster_models = {}
    unique_clusters = sorted(train_agg['Cluster'].dropna().unique())

    for cluster_id in tqdm(unique_clusters, desc="Training"):
        df_cluster = train_agg[train_agg['Cluster'] == cluster_id]
        
        m = Prophet(
            changepoint_prior_scale=0.05, 
            uncertainty_samples=0, 
            daily_seasonality=False,
            weekly_seasonality=False, # We are using weekly data
            yearly_seasonality=True
        )
        m.add_country_holidays(country_name='UK')
        
        # Explicitly register external regressors with Prophet
        for reg in regressors:
            m.add_regressor(reg)
            
        m.fit(df_cluster)
        cluster_models[cluster_id] = m
        
    return cluster_models



def predict_models(cluster_models, test_agg, test_raw, regressors):
    print("Generating forecasts and un-scaling to raw Quantity...")
    
    # 1. Cluster-level Forecast (Vectorized)
    all_cluster_forecasts = []
    for cluster_id, model in cluster_models.items():
        df_test_c = test_agg[test_agg['Cluster'] == cluster_id]
        if len(df_test_c) > 0:
            # Future dataframe MUST contain the 'ds' column and all added regressors
            future = df_test_c[['ds'] + regressors].copy()
            forecast = model.predict(future)
            
            fcst_df = pd.DataFrame({
                'profile_cluster_id': cluster_id,
                'Week': forecast['ds'],
                'Predicted_Quantity_Scaled': forecast['yhat']
            })
            all_cluster_forecasts.append(fcst_df)

    if len(all_cluster_forecasts) == 0:
        return test_raw
        
    global_forecasts = pd.concat(all_cluster_forecasts, ignore_index=True)

    # 2. Merge to Individual SKUs
    test_raw = test_raw.drop(columns=['Predicted_Quantity_Scaled'], errors='ignore')
    test_raw = test_raw.merge(global_forecasts, on=['profile_cluster_id', 'Week'], how='left')
    
    # 3. Inverse Scaling (expm1)
    print("Inverse transforming predictions (expm1)...")
    pred_scaled = test_raw['Predicted_Quantity_Scaled'].fillna(0).values
    pred_scaled = np.clip(pred_scaled, a_min=None, a_max=20.0) # Prevent overflow
    unscaled = np.expm1(pred_scaled)
    
    test_raw['Predicted_Qty'] = np.maximum(unscaled, 0)
    test_raw['Actual_Qty'] = test_raw['Quantity']
                
    return test_raw



def evaluate_models(test_raw):
    print("\nEvaluating Portfolio Performance...")
    
    test_raw['Cluster'] = test_raw['profile_cluster_id']
    test_raw['Date'] = test_raw['Week']
    
    # CRITICAL FIX: Do NOT use .sum() to avoid variance pooling!
    # Pass item-level data directly to accurately compute WMAPE and Median MAPE.
    portfolio_eval = test_raw.dropna(subset=['Actual_Qty', 'Predicted_Qty'])[['Cluster', 'StockCode', 'Date', 'Actual_Qty', 'Predicted_Qty']].copy()
    
    summary = compute_cluster_metrics(portfolio_eval)

    return portfolio_eval, summary



def save_artifacts(cluster_models, regressors, sku_clusters, artifacts_dir="../agent/artifacts"):
    print(f"Saving Cluster Prophet artifacts to {artifacts_dir}...")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    file_name = "prophet_cluster_models.pkl"
    path = os.path.join(artifacts_dir, file_name)
    
    artifact = {
        "cluster_models": cluster_models,
        "regressors": list(regressors), # Equivalent to feature_cols for Prophet
        "sku_clusters": {k: v for k, v in sku_clusters.items()}
    }
    
    joblib.dump(artifact, path)
    print(f"Successfully saved {path}")


def run_prophet_pipeline(file_path, plot=False):
    df_long = load_processed_data(file_path)
    
    train_agg, test_agg, test_raw, regressors = preprocess_and_split(df_long)
    cluster_models = train_models(train_agg, regressors)
    
    test_raw = predict_models(cluster_models, test_agg, test_raw, regressors)
    cluster_eval, summary = evaluate_models(test_raw)
    
    # Extract the SKU mapping dictionary
    sku_clusters = df_long.drop_duplicates(subset=['StockCode']).set_index('StockCode')['profile_cluster_id'].to_dict()
    
    # Save artifacts with all necessary production context
    save_artifacts(cluster_models, regressors, sku_clusters)
    
    if plot:
        plot_cluster_portfolio(cluster_eval, summary, model_label="Prophet (Yearly + Regressors)")
        analyze_time_periods(test_raw)
        
    return cluster_models, test_raw, cluster_eval, summary


if __name__ == "__main__":
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed_retail_data.parquet")
    _, _, _, summary = run_prophet_pipeline(DATA_PATH, plot=False)
    print("\n=== Prophet Evaluation Summary ===")
    print(summary.to_markdown())