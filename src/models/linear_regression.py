import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import joblib

# Ensure project root is in sys.path for absolute imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.tools.evaluation import compute_cluster_metrics
from src.tools.visualization import plot_cluster_portfolio, analyze_time_periods

def load_processed_data(file_path):
    print("Loading processed data...")
    return pd.read_parquet(file_path)

def preprocess_and_split(df_long):
    print("Feature Engineering and Train/Test Split...")
    
    # We rely on the test_cutoff used in process_data.py
    TEST_CUTOFF = pd.to_datetime("2011-09-01")
    
    train = df_long[df_long['Week'] < TEST_CUTOFF].copy()
    test  = df_long[df_long['Week'] >= TEST_CUTOFF].copy()
    
    train = train.sort_values(by=['StockCode', 'Week'])
    test  = test.sort_values(by=['StockCode', 'Week'])

    # Features to scale per SKU
    scale_cols = ['Quantity'] + [f'lag_{l}' for l in (1, 2, 4, 8, 13, 26, 52)] + \
                 [f'rmean_{w}' for w in (4, 13, 26)] + [f'rstd_{w}' for w in (4, 13, 26)] + \
                 ['price_weekly', 'price_percent_change', 'qty_returned']
                 
    # Ensure they exist
    scale_cols = [c for c in scale_cols if c in train.columns]

    # Create empty scaled columns
    for col in scale_cols:
        train[f'{col}_Scaled'] = np.nan
        test[f'{col}_Scaled'] = np.nan

    sku_scalers = {}

    for sku in tqdm(df_long['StockCode'].unique(), desc="Scaling SKUs"):
        scaler = MinMaxScaler()
        train_mask = train['StockCode'] == sku
        test_mask  = test['StockCode'] == sku

        if not train_mask.any():
            print(f"Warning: SKU {sku} has no data in the train set. Skipping...")
            continue
            
        # Fit on train, transform train and test
        # Note: MinMaxScaler scales each column independently. We can scale all columns at once for a given SKU!
        train_vals = train.loc[train_mask, scale_cols].values
        # Handle potential NaNs before fitting
        train_vals = np.nan_to_num(train_vals)
        
        scaled_train = scaler.fit_transform(train_vals)
        
        for i, col in enumerate(scale_cols):
            train.loc[train_mask, f'{col}_Scaled'] = scaled_train[:, i]

        if test_mask.any():
            test_vals = test.loc[test_mask, scale_cols].values
            test_vals = np.nan_to_num(test_vals)
            scaled_test = scaler.transform(test_vals)
            for i, col in enumerate(scale_cols):
                test.loc[test_mask, f'{col}_Scaled'] = scaled_test[:, i]

        sku_scalers[sku] = scaler

    # Drop rows that have NaN in scaled target (should be handled by nan_to_num though, but just in case)
    train = train.dropna(subset=['Quantity_Scaled'])

    # Columns to drop for X
    cols_to_drop = ['Week', 'StockCode', 'Quantity_Scaled'] + scale_cols + ['desc_canonical', 'embedding', 'semantic_cluster_name', 'volume_tier']
    cols_to_drop_train = [c for c in cols_to_drop if c in train.columns]
    
    # We also don't train on clusters that we use to segment
    X_train = train.drop(columns=cols_to_drop_train)
    y_train = train['Quantity_Scaled']

    test = test.sort_values(by=['StockCode', 'Week'])
    X_test = test.drop(columns=cols_to_drop_train)

    print(f"Training shape: {X_train.shape}")
    print(f"Testing shape:  {X_test.shape}")

    # We exclude profile_cluster_id from feature_cols because we segment on it
    feature_cols = X_train.drop(columns=['profile_cluster_id'], errors='ignore').columns.tolist()

    return train, test, X_train, y_train, X_test, sku_scalers, feature_cols


def train_models(X_train, y_train, train):
    print("Training Linear Regression models per Seasonal Profile Cluster...")
    cluster_models = {}
    
    # We use profile_cluster_id
    if 'profile_cluster_id' not in train.columns:
        print("Error: profile_cluster_id not found!")
        return {}
        
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

        model = LinearRegression()
        model.fit(X_train_cluster_vals, y_train_cluster)
        cluster_models[cluster_id] = model
        print(f" - Model for Cluster {int(cluster_id)} trained on {len(X_train_cluster)} historical rows.")

    return cluster_models


def predict_models(cluster_models, test, X_test, sku_scalers):
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
    # For MinMaxScaler, 0 scaled is 0 if minimum quantity was 0. 
    # Usually the minimum quantity is 0 because of zero-filling.
    # So we can just cap the *scaled* prediction at 0.
    test['Predicted_Quantity_Scaled'] = np.maximum(test['Predicted_Quantity_Scaled'].fillna(0), 0)

    print("Predictions Complete!")
    return test


def evaluate_models(test, sku_scalers, train):
    print("\nEvaluating model (raw Quantity)...")
    
    # Reconstruct the scale_cols index to know where 'Quantity' is
    scale_cols = ['Quantity'] + [f'lag_{l}' for l in (1, 2, 4, 8, 13, 26, 52)] + \
                 [f'rmean_{w}' for w in (4, 13, 26)] + [f'rstd_{w}' for w in (4, 13, 26)] + \
                 ['price_weekly', 'price_percent_change', 'qty_returned']
    scale_cols = [c for c in scale_cols if c in train.columns]
    qty_idx = scale_cols.index('Quantity')

    for sku in test['StockCode'].unique():
        if sku not in sku_scalers:
            continue

        sku_mask = test['StockCode'] == sku
        sku_data = test[sku_mask].copy()

        valid = sku_data['Quantity'].notna() & sku_data['Predicted_Quantity_Scaled'].notna()
        if valid.sum() == 0:
            continue

        y_true_qty = sku_data.loc[valid, 'Quantity'].values
        scaler = sku_scalers[sku]
        
        # We need to construct a dummy array to inverse_transform, because scaler was fit on all scale_cols
        dummy_arr = np.zeros((len(y_true_qty), len(scale_cols)))
        dummy_arr[:, qty_idx] = sku_data.loc[valid, 'Predicted_Quantity_Scaled'].values
        
        y_pred_qty = scaler.inverse_transform(dummy_arr)[:, qty_idx]

        # Cap predictions at 0 real units
        y_pred_qty = np.maximum(y_pred_qty, 0)
        
        test.loc[sku_data.index[valid], 'Actual_Qty'] = y_true_qty
        test.loc[sku_data.index[valid], 'Predicted_Qty'] = y_pred_qty

    # Group by profile_cluster_id and Date (Week)
    test['Cluster'] = test['profile_cluster_id']
    test['Date'] = test['Week']
    
    cluster_eval = (
        test.dropna(subset=['Actual_Qty', 'Predicted_Qty'])
        .groupby(['Cluster', 'Date'], observed=True)[['Actual_Qty', 'Predicted_Qty']]
        .sum()
        .reset_index()
    )

    summary = compute_cluster_metrics(cluster_eval)

    return cluster_eval, summary


def save_cluster_artifacts(cluster_models, sku_scalers, feature_cols, sku_clusters, artifacts_dir="../agent/artifacts"):
    print(f"Saving Cluster Linear Regression artifacts to {artifacts_dir}...")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    artifact = {
        "cluster_models": cluster_models,
        "sku_scalers": sku_scalers,
        "feature_cols": list(feature_cols),
        "sku_clusters": {k: v for k, v in sku_clusters.items()}
    }
    
    path = os.path.join(artifacts_dir, "lr_cluster_models.pkl")
    joblib.dump(artifact, path)
    print(f"Successfully saved {path}")


def run_linear_regression_pipeline(file_path, plot=False):
    """
    Complete pipeline to load data, train models, predict, evaluate, and visualize results.
    """
    df_long = load_processed_data(file_path)
    train, test, X_train, y_train, X_test, sku_scalers, feature_cols = preprocess_and_split(df_long)
    cluster_models = train_models(X_train, y_train, train)
    test = predict_models(cluster_models, test, X_test, sku_scalers)
    cluster_eval, summary = evaluate_models(test, sku_scalers, train)
    
    sku_clusters = df_long.drop_duplicates(subset=['StockCode']).set_index('StockCode')['profile_cluster_id'].to_dict()
    save_cluster_artifacts(cluster_models, sku_scalers, feature_cols, sku_clusters, artifacts_dir=os.path.join(os.path.dirname(__file__), '..', '..', 'agent', 'artifacts'))
    
    if plot:
        plot_cluster_portfolio(cluster_eval, summary)
        analyze_time_periods(test)
    
    return cluster_models, test, cluster_eval, summary

if __name__ == "__main__":
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed_retail_data.parquet")
    run_linear_regression_pipeline(DATA_PATH, plot=False)