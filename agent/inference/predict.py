"""
agent/inference/predict.py
--------------------------
Robust inference module for LR and Prophet models.
Adapted for Retail Demand Forecasting.
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys
import json
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
    """Unified inference engine for Retail products. Supports 'auto' mode."""
    model_name = model_name.lower()
    
    # Auto-routing: load the best model lookup and resolve
    if model_name == 'auto':
        lookup_path = os.path.join(ARTIFACTS_DIR, "best_model_per_sku.json")
        if os.path.exists(lookup_path):
            with open(lookup_path, 'r') as f:
                lookup = json.load(f)
            entry = lookup.get(stock_code, {})
            model_name = entry.get("model", "lgb")  # fallback to lgb
        else:
            model_name = "lgb"  # fallback if no selection has been run
    
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
        if model_name in ['lr', 'lgb']:
            feature_cols = art["feature_cols"]

            # 1. Global Scaling — log1p for volumes, MinMax for ratios.
            # Use the scaler saved during training so the range matches exactly.
            vol_features = [f'lag_{l}' for l in (1, 4, 13, 52)] + [f'rmean_{w}' for w in (4, 13)] + [f'rstd_{w}' for w in (4, 13)]
            for col in vol_features:
                if col in test_df.columns:
                    test_df[f'{col}_Scaled'] = np.log1p(test_df[col].clip(lower=0).fillna(0))

            ratio_features = ['return_rate_4w', 'return_rate_13w', 'price_weekly', 'price_percent_change', 'ADI', 'CV2', 'share_zero_weeks']
            ratio_features = [c for c in ratio_features if c in test_df.columns]

            if ratio_features:
                scaler = art.get("global_scaler")
                if scaler is None:
                    # Fallback for artifacts saved before the scaler was persisted.
                    # Re-fit on training rows only to avoid leaking test-period stats.
                    from sklearn.preprocessing import MinMaxScaler
                    scaler = MinMaxScaler()
                    train_rows = df_all[df_all['Week'] < TEST_CUTOFF_DT]
                    scaler.fit(train_rows[ratio_features].fillna(0))
                test_df[[f"{c}_Scaled" for c in ratio_features]] = scaler.transform(test_df[ratio_features].fillna(0))

            # 2. Model Specific Inference
            if model_name == 'lr':
                # One-Hot Encode before reindexing so dummy columns are correctly populated.
                cat_cols = ['volume_tier', 'semantic_cluster_name', 'demand_class']
                test_df_lr = pd.get_dummies(test_df, columns=[c for c in cat_cols if c in test_df.columns], drop_first=False)
                X = test_df_lr.reindex(columns=feature_cols, fill_value=0)
                preds_scaled = model_obj.predict(X.astype(float).values)

            elif model_name == 'lgb':
                # Bulletproof Categorical Casting to prevent C++ Segmentation Faults
                X = pd.DataFrame(index=test_df.index)
                cat_cols = ['volume_tier', 'semantic_cluster_name', 'demand_class']

                for col in feature_cols:
                    if col in cat_cols:
                        exact_categories = df_all[col].astype('category').cat.categories
                        X[col] = pd.Categorical(test_df[col], categories=exact_categories, ordered=False).codes.astype(float)
                    elif col in test_df.columns:
                        X[col] = test_df[col].astype(float)
                    else:
                        val_assigned = False
                        for cat in cat_cols:
                            if col.startswith(cat + "_"):
                                expected_val = col[len(cat) + 1:]
                                X[col] = (test_df[cat].astype(str) == expected_val).astype(float)
                                val_assigned = True
                                break
                        if not val_assigned:
                            X[col] = 0.0

                # num_threads=1 prevents OpenMP crashes in LangChain.
                preds_scaled = model_obj.predict(X.values, num_threads=1)

        elif model_name == 'prophet':
            future_df = test_df.copy()
            future_df['ds'] = future_df['Week']

            regressors = art.get("regressors", [])
            for reg in regressors:
                if reg not in future_df.columns:
                    future_df[reg] = 0.0

            future = future_df[['ds'] + regressors]
            forecast = model_obj.predict(future)
            preds_scaled = forecast['yhat'].values

        elif model_name == 'nst':
            # NST is a global multivariate model: rebuild from state_dict, run on
            # all cluster SKUs, then slice the output for the requested SKU.
            import torch
            from src.models.ns_transformer.architecture import NonStationaryTransformer
            from src.models.ns_transformer.train import predict_ns_transformer, DEFAULT

            nst_art = _load_artifact('nst')
            nst_cluster_data = nst_art["cluster_models"].get(cluster_id)
            if nst_cluster_data is None:
                raise KeyError(f"NST cluster model not found for cluster '{cluster_id}'.")

            meta = nst_cluster_data["meta"]
            p = {**DEFAULT, **meta.get("params", {})}
            skus_in_cluster = nst_cluster_data["skus"]
            n_skus = meta["panel_shape"][1]

            nst_model = NonStationaryTransformer(
                enc_in=n_skus, c_out=n_skus,
                seq_len=p["SEQ_LEN"], label_len=p["LABEL_LEN"], pred_len=p["PRED_LEN"],
                d_model=p["D_MODEL"], n_heads=p["N_HEADS"],
                e_layers=p["E_LAYERS"], d_layers=p["D_LAYERS"],
                d_ff=p["D_FF"], dropout=p["DROPOUT"],
                n_time_feat=p.get("N_TIME_FEAT", 4),
            )
            nst_model.load_state_dict(nst_cluster_data["model_state"])

            fcst_df = predict_ns_transformer(
                nst_model, df_all, skus_in_cluster,
                horizon=horizon_weeks,
                seq_len=p["SEQ_LEN"], label_len=p["LABEL_LEN"],
                device=torch.device("cpu"),
            )
            sku_fcst = fcst_df[fcst_df["StockCode"] == stock_code].sort_values("Horizon")
            preds_qty = [round(max(0.0, float(v)), 2) for v in sku_fcst["Forecast"].values[:horizon_weeks]]
            # NST already applies expm1 internally — return directly without further inversion.
            return ForecastResult(model_name, stock_code, future_ts, preds_qty)

        elif model_name == 'sarimax':
            regressors = art.get("regressors", [])
            for reg in regressors:
                if reg not in test_df.columns:
                    test_df[reg] = 0.0
            X_test = test_df[regressors].fillna(0).values
            preds_scaled = model_obj.forecast(steps=len(test_df), exog=X_test)

        else:
            raise ValueError(f"Unknown model: {model_name}. Allowed: 'lr', 'prophet', 'lgb', 'nst', 'sarimax'.")
            
        # 5. Inverse Scale target (log1p -> expm1)
        preds_scaled = np.clip(preds_scaled, a_min=None, a_max=20.0) # Prevent overflow
        preds_qty = np.expm1(preds_scaled)
        preds_qty = [round(max(0.0, float(v)), 2) for v in preds_qty]

        return ForecastResult(model_name, stock_code, future_ts, preds_qty)

    except Exception as e:
        return ForecastResult(model_name, stock_code, [], [], error=str(e))