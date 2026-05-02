"""
agent/inference/predict.py
--------------------------
Robust inference module for LR and Prophet models.
Adapted for Retail Demand Forecasting.
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

warnings.filterwarnings("ignore")

# Absolute path setup
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
ARTIFACTS_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "artifacts")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import TEST_CUTOFF_DT

@dataclass
class ForecastResult:
    model_name: str
    stock_code: str
    timestamps: List[str]
    predictions_qty: List[float]
    error: Optional[str] = None

    @property
    def total_qty(self) -> float:
        return round(sum(self.predictions_qty), 0) if self.predictions_qty else 0.0

    @property
    def mean_qty(self) -> float:
        return round(float(np.mean(self.predictions_qty)), 1) if self.predictions_qty else 0.0

    @property
    def peak_qty(self) -> float:
        return round(float(np.max(self.predictions_qty)), 0) if self.predictions_qty else 0.0

    @property
    def peak_timestamp(self) -> str:
        if not self.predictions_qty: return "N/A"
        idx = int(np.argmax(self.predictions_qty))
        return self.timestamps[idx]

    def to_summary(self) -> str:
        if self.error:
            return f"[{self.model_name.upper()}] ERROR for {self.stock_code}: {self.error}"
        return (
            f"--- {self.model_name.upper()} FORECAST | PRODUCT {self.stock_code} ---\n"
            f"Period        : {self.timestamps[0]}  to  {self.timestamps[-1]}\n"
            f"Total Sales   : {self.total_qty:,.0f} units\n"
            f"Average Weekly: {self.mean_qty} units/week\n"
            f"Peak Weekly   : {self.peak_qty} units (Week of {self.peak_timestamp})\n"
        )

# ── HELPERS ──────────────────────────────────────────────────────────────────

def _load_artifact(model: str) -> dict:
    """Loads a cluster-based artifact from agent/artifacts/."""
    filename = f"{model}_cluster_models.pkl"
    path = os.path.join(ARTIFACTS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact not found at {path}")
    return joblib.load(path)


def predict_retail(stock_code: str, model_name: str, df_all: pd.DataFrame, horizon_weeks: int = 4) -> ForecastResult:
    """Unified inference engine for Retail products."""
    model_name = model_name.lower()
    
    try:
        # 1. Fetch Cluster ID
        product_data = df_all[df_all["StockCode"] == stock_code]
        if product_data.empty:
            raise KeyError(f"StockCode '{stock_code}' not found in dataset.")
            
        cluster_id = product_data["profile_cluster_id"].iloc[0]

        # 2. Extract testing segment for this product
        test_df = product_data[product_data['Week'] >= TEST_CUTOFF_DT].copy()
        test_df = test_df.sort_values(by="Week").head(horizon_weeks)
        
        if test_df.empty:
            raise ValueError(f"No testing data available for '{stock_code}' after {TEST_CUTOFF_DT.strftime('%Y-%m-%d')}.")
            
        future_ts = test_df['Week'].dt.strftime('%Y-%m-%d').tolist()

        # 3. Load Artifact
        art = _load_artifact(model_name)
        cluster_models = art["cluster_models"]
        
        if cluster_id not in cluster_models:
             raise KeyError(f"Cluster model not found for cluster '{cluster_id}'.")
             
        model_obj = cluster_models[cluster_id]

        # 4. Model Inference
        if model_name == 'lr':
            sku_scalers = art["sku_scalers"]
            feature_cols = art["feature_cols"]
            
            if stock_code not in sku_scalers:
                raise KeyError(f"StockCode '{stock_code}' has no trained feature scaler.")
            scaler = sku_scalers[stock_code]
            
            # Scale features dynamically
            scale_cols = [f'lag_{l}' for l in (1, 2, 4, 8, 13, 26, 52)] + \
                         [f'rmean_{w}' for w in (4, 13, 26)] + [f'rstd_{w}' for w in (4, 13, 26)] + \
                         ['price_weekly', 'price_percent_change', 'qty_returned']
            scale_cols = [c for c in scale_cols if c in test_df.columns]
            
            # Create a copy with scaled features
            scaled_test_df = test_df.copy()
            if scale_cols:
                scaled_test_df[[f"{c}_Scaled" for c in scale_cols]] = scaler.transform(test_df[scale_cols])
            
            # Reindex to exact feature columns
            X = scaled_test_df.reindex(columns=feature_cols, fill_value=0).astype(float).values
            
            # Predict
            preds_scaled = model_obj.predict(X)
            
        elif model_name == 'prophet':
            future_df = test_df.copy()
            future_df['ds'] = future_df['Week']
            # Make sure holiday regressors are included
            forecast = model_obj.predict(future_df)
            preds_scaled = forecast['yhat'].values
            
        else:
            raise ValueError(f"Unknown model: {model_name}. Allowed: 'lr', 'prophet'.")

        # 5. Inverse Scale target (log1p -> expm1)
        preds_scaled = np.clip(preds_scaled, a_min=None, a_max=20.0) # Prevent overflow
        preds_qty = np.expm1(preds_scaled)
        preds_qty = [round(max(0.0, float(v)), 2) for v in preds_qty]

        return ForecastResult(model_name, stock_code, future_ts, preds_qty)

    except Exception as e:
        return ForecastResult(model_name, stock_code, [], [], error=str(e))