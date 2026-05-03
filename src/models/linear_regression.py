import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import joblib

# Ensure project root is in sys.path for absolute imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import TEST_CUTOFF_DT
from src.tools import compute_cluster_metrics, load_processed_data, plot_cluster_portfolio, analyze_time_periods


def preprocess_and_split(df_long):
    print("Feature Engineering and Train/Test Split...")
    
    cat_cols = ['volume_tier', 'semantic_cluster_name', 'demand_class']
    for col in cat_cols:
        if col in df_long.columns:
            df_long[col] = df_long[col].astype(str)
            
    df_long = pd.get_dummies(df_long, columns=[c for c in cat_cols if c in df_long.columns], drop_first=False)
    
    dummy_cols = [c for c in df_long.columns if any(c.startswith(f"{cat}_") for cat in cat_cols)]
    df_long[dummy_cols] = df_long[dummy_cols].astype(float)
    
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
    
    # 1. Log-transform volume-based features to match target scale and prevent outlier explosions
    vol_features = [f'lag_{l}' for l in (1, 4, 13, 52)] + [f'rmean_{w}' for w in (4, 13)] + [f'rstd_{w}' for w in (4, 13)]
    for col in vol_features:
        if col in train.columns:
            train[f'{col}_Scaled'] = np.log1p(train[col].clip(lower=0).fillna(0))
            test[f'{col}_Scaled']  = np.log1p(test[col].clip(lower=0).fillna(0))

    # 2. Global MinMax for ratios, prices, and static metrics (Fast & Stable)
    ratio_features = ['return_rate_4w', 'return_rate_13w', 'price_weekly', 'price_percent_change', 'ADI', 'CV2', 'share_zero_weeks']
    ratio_features = [c for c in ratio_features if c in train.columns]
    
    global_scaler = MinMaxScaler()
    
    # Fill NaNs before scaling
    train[ratio_features] = train[ratio_features].fillna(0)
    test[ratio_features]  = test[ratio_features].fillna(0)
    
    train_scaled = global_scaler.fit_transform(train[ratio_features])
    test_scaled  = global_scaler.transform(test[ratio_features])
    
    for i, col in enumerate(ratio_features):
        train[f'{col}_Scaled'] = train_scaled[:, i]
        test[f'{col}_Scaled']  = test_scaled[:, i]

    # Drop rows that have NaN in scaled target (should be handled by nan_to_num though, but just in case)
    train = train.dropna(subset=['Quantity_Scaled'])

    # Columns to drop for X
    cols_to_drop = [
        'Week', 'StockCode', 'Quantity', 'Quantity_Scaled', 'qty_returned', 'Revenue',
        'week_of_year', 'month', 'quarter', 'year',
        'price_median', 'mean_basket_size', 'n_unique_customers', 'country_uk_share'
    ] + scale_cols + ['desc_canonical', 'embedding', 'semantic_cluster_name', 'volume_tier']
    
    cols_to_drop_train = [c for c in cols_to_drop if c in train.columns]
    
    # We also don't train on clusters that we use to segment
    X_train = train.drop(columns=cols_to_drop_train)
    y_train = train['Quantity_Scaled']

    test = test.sort_values(by=['StockCode', 'Week'])
    X_test = test.drop(columns=cols_to_drop_train)

    print(f"Training shape: {X_train.shape}")
    print(f"Testing shape:  {X_test.shape}")
    print(f"Features:  {X_test.columns.tolist()}")

    # We exclude profile_cluster_id from feature_cols because we segment on it
    feature_cols = X_train.drop(columns=['profile_cluster_id'], errors='ignore').columns.tolist()

    return train, test, X_train, y_train, X_test, feature_cols


def train_models(X_train, y_train, train):
    print("Training Linear Regression models per Seasonal Profile Cluster...")
    cluster_models = {}
        
    unique_clusters = train['profile_cluster_id'].dropna().unique()

    for cluster_id in sorted(unique_clusters):
        cluster_mask = train['profile_cluster_id'] == cluster_id
        X_train_cluster = X_train[cluster_mask].drop(columns=['profile_cluster_id'], errors='ignore')
        
        # Fill remaining NaNs if any (e.g. from joining clusters that had NaNs)
        X_train_cluster = X_train_cluster.fillna(0)
        X_train_cluster_vals = np.nan_to_num(X_train_cluster.values, nan=0.0, posinf=0.0, neginf=0.0)
        y_train_cluster = y_train[cluster_mask]

        if len(X_train_cluster) == 0:
            continue

        model = Ridge(alpha=1.0)
        model.fit(X_train_cluster_vals, y_train_cluster)
        cluster_models[cluster_id] = model
        print(f" - Model for Cluster {int(cluster_id)} trained on {len(X_train_cluster)} historical rows.")

    return cluster_models


def predict_models(cluster_models, test, X_test):
    print("Predicting on Test Set...")
    test['Predicted_Quantity_Scaled'] = np.nan

    for cluster_id, model in cluster_models.items():
        cluster_mask = test['profile_cluster_id'] == cluster_id
        X_test_cluster = X_test[cluster_mask].drop(columns=['profile_cluster_id'], errors='ignore')
        X_test_cluster = X_test_cluster.fillna(0)
        X_test_cluster_vals = np.nan_to_num(X_test_cluster.values, nan=0.0, posinf=0.0, neginf=0.0)
        
        if len(X_test_cluster) > 0:
            preds = model.predict(X_test_cluster_vals)
            test.loc[cluster_mask, 'Predicted_Quantity_Scaled'] = preds

    print("Applying physical constraints (Capping at 0)...")
    # log1p transformation means inverse is expm1.
    # We will do the inverse transform in evaluate_models, but let's cap the scaled predictions here just in case.
    test['Predicted_Quantity_Scaled'] = np.maximum(test['Predicted_Quantity_Scaled'].fillna(0), 0)

    print("Predictions Complete!")
    return test


def evaluate_models(test):
    print("\nEvaluating model (raw Quantity)...")
    for sku in test['StockCode'].unique():
        sku_mask = test['StockCode'] == sku
        sku_data = test[sku_mask].copy()

        valid = sku_data['Quantity'].notna() & sku_data['Predicted_Quantity_Scaled'].notna()
        if valid.sum() == 0:
            continue

        y_true_qty = sku_data.loc[valid, 'Quantity'].values
        y_pred_scaled = sku_data.loc[valid, 'Predicted_Quantity_Scaled'].values
        y_pred_scaled = np.clip(y_pred_scaled, a_min=None, a_max=20.0) # max ~485M units
        y_pred_qty = np.expm1(y_pred_scaled)

        # Cap predictions at 0 real units
        y_pred_qty = np.maximum(y_pred_qty, 0)
        
        test.loc[sku_data.index[valid], 'Actual_Qty'] = y_true_qty
        test.loc[sku_data.index[valid], 'Predicted_Qty'] = y_pred_qty

    # Group by profile_cluster_id and Date (Week)
    test['Cluster'] = test['profile_cluster_id']
    test['Date'] = test['Week']
    
    # Pass the raw, item-level predictions to accurately calculate Median MAPE and WMAPE without "variance pooling".
    cluster_eval = test.dropna(subset=['Actual_Qty', 'Predicted_Qty'])[['Cluster', 'StockCode', 'Date', 'Actual_Qty', 'Predicted_Qty']].copy()

    summary = compute_cluster_metrics(cluster_eval)

    return cluster_eval, summary


def save_artifacts(cluster_models, feature_cols, sku_clusters, artifacts_dir="../agent/artifacts"):
    print(f"Saving Cluster Linear Regression artifacts to {artifacts_dir}...")
    os.makedirs(artifacts_dir, exist_ok=True)

    file_name = "lr_cluster_models.pkl"
    path = os.path.join(artifacts_dir, file_name)
    
    artifact = {
        "cluster_models": cluster_models,
        "feature_cols": list(feature_cols),
        "sku_clusters": {k: v for k, v in sku_clusters.items()}
    }
    
    joblib.dump(artifact, path)
    print(f"Successfully saved {path}")


def run_linear_regression_pipeline(file_path, plot=False):
    """
    Complete pipeline to load data, train models, predict, evaluate, and visualize results.
    """
    df_long = load_processed_data(file_path)
    train, test, X_train, y_train, X_test, feature_cols = preprocess_and_split(df_long)
    cluster_models = train_models(X_train, y_train, train)
    test = predict_models(cluster_models, test, X_test)
    cluster_eval, summary = evaluate_models(test)
    
    sku_clusters = df_long.drop_duplicates(subset=['StockCode']).set_index('StockCode')['profile_cluster_id'].to_dict()
    save_artifacts(cluster_models, feature_cols, sku_clusters, artifacts_dir=os.path.join(PROJECT_ROOT, 'agent', 'artifacts'))

    if plot:
        plot_cluster_portfolio(cluster_eval, summary)
        analyze_time_periods(test)
    
    return cluster_models, test, cluster_eval, summary

if __name__ == "__main__":
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed_retail_data.parquet")
    _, _, _, summary = run_linear_regression_pipeline(DATA_PATH, plot=False)
    print("\n=== Linear Regression Evaluation Summary ===")
    print(summary.to_markdown())