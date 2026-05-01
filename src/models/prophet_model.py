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

from src.tools.evaluation import compute_cluster_metrics
from src.tools.visualization import plot_cluster_portfolio, analyze_time_periods

# Suppress Prophet/cmdstanpy verbose logging
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)

def load_processed_data(file_path):
    print("Loading processed data...")
    return pd.read_parquet(file_path)

def preprocess_and_split(df_long):
    print("Preparing train/test split and log-scaling...")
    
    TEST_CUTOFF = pd.to_datetime("2011-09-01")
    
    train_raw = df_long[df_long['Week'] < TEST_CUTOFF].copy()
    test_raw  = df_long[df_long['Week'] >= TEST_CUTOFF].copy()
    
    # 1. Log Scaling for Quantity
    print("Applying log1p scaling to Quantity...")
    train_raw['Quantity_Scaled'] = np.log1p(train_raw['Quantity'])
    test_raw['Quantity_Scaled'] = np.log1p(test_raw['Quantity'])

    # 2. Aggregation by Cluster for Prophet
    print("Aggregating data by Cluster for Prophet training...")
    if 'profile_cluster_id' not in train_raw.columns:
        raise ValueError("Error: profile_cluster_id not found in data!")

    train_agg = train_raw.groupby(['profile_cluster_id', 'Week'], observed=True)['Quantity_Scaled'].mean().reset_index()
    test_agg  = test_raw.groupby(['profile_cluster_id', 'Week'], observed=True)['Quantity_Scaled'].mean().reset_index()
    
    train_agg = train_agg.rename(columns={'Week': 'ds', 'Quantity_Scaled': 'y', 'profile_cluster_id': 'Cluster'})
    test_agg  = test_agg.rename(columns={'Week': 'ds', 'Quantity_Scaled': 'y', 'profile_cluster_id': 'Cluster'})
    
    return train_agg, test_agg, test_raw

def train_models(train_agg):
    print(f"Training Prophet models for {train_agg['Cluster'].nunique()} clusters...")
    cluster_models = {}
    unique_clusters = sorted(train_agg['Cluster'].dropna().unique())

    for cluster_id in tqdm(unique_clusters, desc="Training"):
        df_cluster = train_agg[train_agg['Cluster'] == cluster_id]
        
        m = Prophet(
            changepoint_prior_scale=0.05, 
            uncertainty_samples=0, 
            daily_seasonality=False,
            weekly_seasonality=False, # We are using weekly data, so weekly seasonality is not possible
            yearly_seasonality=True
        )
        m.add_country_holidays(country_name='UK')
        m.fit(df_cluster)
        cluster_models[cluster_id] = m
        
    return cluster_models

def predict_models(cluster_models, test_agg, test_raw):
    print("Generating forecasts and un-scaling to raw Quantity...")
    
    # 1. Cluster-level Forecast (Vectorized)
    all_cluster_forecasts = []
    for cluster_id, model in cluster_models.items():
        df_test_c = test_agg[test_agg['Cluster'] == cluster_id]
        if len(df_test_c) > 0:
            future = df_test_c[['ds']].copy()
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
    
    # Portfolio aggregation by Cluster and Date
    test_raw['Cluster'] = test_raw['profile_cluster_id']
    test_raw['Date'] = test_raw['Week']
    
    portfolio_eval = (
        test_raw.dropna(subset=['Actual_Qty', 'Predicted_Qty'])
        .groupby(['Cluster', 'Date'], observed=True)[['Actual_Qty', 'Predicted_Qty']]
        .sum()
        .reset_index()
    )
    
    summary = compute_cluster_metrics(portfolio_eval)

    return portfolio_eval, summary

def save_prophet_artifacts(cluster_models, artifacts_dir=None):
    if artifacts_dir is None:
        artifacts_dir = os.path.join(PROJECT_ROOT, 'agent', 'artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)
    
    file_name = "prophet_cluster_models.pkl"
    path = os.path.join(artifacts_dir, file_name)
    
    artifact = {
        "cluster_models": cluster_models,
        "is_retail": True
    }
    joblib.dump(artifact, path)
    print(f"\n Prophet artifacts successfully saved to: {path}")

def run_prophet_pipeline(file_path, plot=False):
    df_long = load_processed_data(file_path)
    
    train_agg, test_agg, test_raw = preprocess_and_split(df_long)
    cluster_models = train_models(train_agg)
    
    test_raw = predict_models(cluster_models, test_agg, test_raw)
    portfolio_eval, summary = evaluate_models(test_raw)
    
    save_prophet_artifacts(cluster_models)
    
    if plot:
        plot_cluster_portfolio(portfolio_eval, summary, model_label="Prophet (Yearly)")
        analyze_time_periods(test_raw)
        
    return cluster_models, test_raw, portfolio_eval, summary

if __name__ == "__main__":
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed_retail_data.parquet")
    _, _, _, summary = run_prophet_pipeline(DATA_PATH, plot=False)
    print("\n=== Prophet Evaluation Summary ===")
    print(summary.to_markdown())