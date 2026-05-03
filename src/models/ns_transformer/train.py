"""
Training + inference pipeline for the Non-Stationary Transformer on the
weekly SKU panel. This is a SKELETON: glue between the architecture and the
project's data layout. Heavy lifting (full training loop, early stopping,
device management) is here in compact form — extend with your hyperparameter
sweep when you go to production.
"""
from pathlib import Path
import gc
import numpy as np
import pandas as pd
import os
import joblib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.config import TEST_CUTOFF_DT
from src.tools.evaluation import compute_cluster_metrics
from src.tools.visualization import plot_cluster_portfolio, analyze_time_periods
from src.tools import load_processed_data

from .architecture import NonStationaryTransformer

DEFAULT = {
    # --- Context & Horizon ---
    "SEQ_LEN": 16,      # Shorter lookback = more sliding windows for training
    "LABEL_LEN": 12,    # Overlap between encoder and decoder to provide context
    "PRED_LEN": 12,     # The target 12-week forecast horizon

    # --- Architecture Capacity (Right-sized for ~20-40 sliding windows) ---
    "D_MODEL": 64,      # Drastically reduced to match data complexity
    "N_HEADS": 4,       # Attention heads
    "E_LAYERS": 2,      # Shallower encoder to avoid overfitting
    "D_LAYERS": 1,      # Single decoder layer
    "D_FF": 128,        # Feed-forward dimension (2x D_MODEL)
    "DROPOUT": 0.3,     # Strong dropout to fight overfitting on tiny dataset

    # --- Training Dynamics ---
    "BATCH_SIZE": 4,    # Smaller batches = more gradient updates per epoch
    "LR": 5e-4,         # Higher initial LR — cosine scheduler will decay it
    "WEIGHT_DECAY": 0.01,  # L2 regularization to prevent weight explosion
    "EPOCHS": 200,      # Sufficient with cosine annealing
    "PATIENCE": 40,     # Early stopping patience (only after MIN_EPOCHS)
    "MIN_EPOCHS": 50,   # Don't check early stopping until the cosine LR has decayed enough
    "AUG_NOISE_STD": 0.05, # Gaussian noise augmentation on training inputs

    # --- Time Features ---
    "N_TIME_FEAT": 4,   # wk_of_year, sin_year, cos_year, month_norm
}


def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():         return torch.device("cuda")
    return torch.device("cpu")


class SkuPanelDataset(Dataset):
    """Sliding-window dataset over a (T, F) numpy array per SKU.
    Time-features are sin/cos of week-of-year; replace with calendar features
    from src/tools/features.py for richer signal."""
    def __init__(self, data: np.ndarray, seq_len: int, label_len: int, pred_len: int):
        self.data = data.astype(np.float32)
        self.seq_len, self.label_len, self.pred_len = seq_len, label_len, pred_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - self.pred_len + 1)

    def _time_features(self, start, end):
        weeks = np.arange(start, end)
        wk_of_year = (weeks % 52) / 51.0 - 0.5
        sin_year = np.sin(2 * np.pi * (weeks % 52) / 52)
        cos_year = np.cos(2 * np.pi * (weeks % 52) / 52)
        month_norm = ((weeks % 52) // 4) / 12.0 - 0.5
        return np.stack([wk_of_year, sin_year, cos_year, month_norm], axis=-1).astype(np.float32)

    def __getitem__(self, idx):
        s_end = idx + self.seq_len
        r_end = s_end + self.pred_len
        return (
            torch.tensor(self.data[idx:s_end]),
            torch.tensor(self.data[s_end - self.label_len : r_end]),
            torch.tensor(self._time_features(idx, s_end)),
            torch.tensor(self._time_features(s_end - self.label_len, r_end)),
        )


def _build_panel(weekly_sku: pd.DataFrame, skus: list[str]) -> np.ndarray:
    """Pivot weekly_sku to a (T, N_skus) matrix on a complete weekly axis, fill 0."""
    pv = (
        weekly_sku[weekly_sku["StockCode"].isin(skus)]
        .pivot(index="Week", columns="StockCode", values="Quantity")
        .reindex(columns=skus)
    )
    full = pd.date_range(pv.index.min(), pv.index.max(), freq="W-MON")
    return pv.reindex(full).fillna(0).values  # (T, N)


def train_ns_transformer(
    weekly_sku: pd.DataFrame,
    skus: list[str],
    params: dict | None = None,
    device: torch.device | None = None,
) -> tuple[NonStationaryTransformer, dict]:
    p = {**DEFAULT, **(params or {})}
    device = device or get_device()

    raw_panel = _build_panel(weekly_sku, skus)         # (T, N_skus)
    
    # 1. CRITICAL: Log1p transformation to stabilize extreme retail Pareto variance
    panel = np.log1p(np.clip(raw_panel, a_min=0, a_max=None))
    
    # 2. STRICT SPLIT: Prevent Data Leakage using the global cutoff
    full_idx = pd.date_range(weekly_sku["Week"].min(), weekly_sku["Week"].max(), freq="W-MON")
    cutoff_idx = full_idx.searchsorted(pd.to_datetime(TEST_CUTOFF_DT))
    
    train_full = panel[:cutoff_idx]
    
    # Strict split with gap: exclude 2*PRED_LEN from training tail so that
    # validation windows have zero overlap with training context.
    gap = 2 * p["PRED_LEN"]
    train_data = train_full[:-gap] if len(train_full) > gap + p["SEQ_LEN"] + p["PRED_LEN"] else train_full[:-p["PRED_LEN"]]
    
    val_size = p["SEQ_LEN"] + p["PRED_LEN"]
    val_data = train_full[-val_size:]

    n_time_feat = p.get("N_TIME_FEAT", 4)

    train_ds = SkuPanelDataset(train_data, p["SEQ_LEN"], p["LABEL_LEN"], p["PRED_LEN"])
    val_ds = SkuPanelDataset(val_data, p["SEQ_LEN"], p["LABEL_LEN"], p["PRED_LEN"])
    
    # drop_last=False: our dataset generates very few sliding windows.
    train_loader = DataLoader(train_ds, batch_size=p["BATCH_SIZE"], shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=p["BATCH_SIZE"], shuffle=False)

    model = NonStationaryTransformer(
        enc_in=panel.shape[1], c_out=panel.shape[1],
        seq_len=p["SEQ_LEN"], label_len=p["LABEL_LEN"], pred_len=p["PRED_LEN"],
        d_model=p["D_MODEL"], n_heads=p["N_HEADS"],
        e_layers=p["E_LAYERS"], d_layers=p["D_LAYERS"],
        d_ff=p["D_FF"], dropout=p["DROPOUT"],
        n_time_feat=n_time_feat,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=p["LR"], weight_decay=p.get("WEIGHT_DECAY", 0.01))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=p["EPOCHS"], eta_min=1e-6)
    crit = nn.HuberLoss()  # Huber ≈ MAE on log1p — better aligned with WMAPE than MSE
    noise_std = p.get("AUG_NOISE_STD", 0.05)

    best, patience_count, best_state = float("inf"), 0, None
    for epoch in range(p["EPOCHS"]):
        model.train()
        for x, y, xm, ym in train_loader:
            x, y, xm, ym = x.to(device), y.to(device), xm.to(device), ym.to(device)
            # Data augmentation: add Gaussian noise to encoder inputs to fight overfitting
            if noise_std > 0:
                x = x + torch.randn_like(x) * noise_std
            yhat = model(x, xm, y, ym)
            loss = crit(yhat, y[:, -p["PRED_LEN"] :, :])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()

        model.eval()
        v = []
        with torch.no_grad():
            for x, y, xm, ym in val_loader:
                x, y, xm, ym = x.to(device), y.to(device), xm.to(device), ym.to(device)
                v.append(crit(model(x, xm, y, ym), y[:, -p["PRED_LEN"] :, :]).item())
        avg = float(np.mean(v)) if v else float("inf")
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1:>2}  val={avg:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if avg < best:
            best, patience_count = avg, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif epoch >= p.get("MIN_EPOCHS", 50):
            patience_count += 1
            if patience_count >= p["PATIENCE"]:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    gc.collect()
    return model, {"skus": skus, "params": p, "panel_shape": panel.shape}


def predict_ns_transformer(
    model: NonStationaryTransformer,
    weekly_sku: pd.DataFrame,
    skus: list[str],
    horizon: int = 12,
    seq_len: int = None,
    label_len: int = None,
    device: torch.device | None = None,
) -> pd.DataFrame:
    device = device or get_device()
    model.eval()
    
    # 1. Dynamically pull the exact parameters the model was compiled with
    if seq_len is None or label_len is None:
        seq_len = DEFAULT["SEQ_LEN"]
        label_len = DEFAULT["LABEL_LEN"]
    
    raw_panel = _build_panel(weekly_sku, skus)
    # 2. Apply Log1p to the historical context to match training inputs
    panel = np.log1p(np.clip(raw_panel, a_min=0, a_max=None))
    
    enc_x = torch.tensor(panel[-seq_len:], dtype=torch.float32).unsqueeze(0).to(device)
    dec_x = torch.tensor(panel[-label_len:], dtype=torch.float32).unsqueeze(0).to(device)
    dec_x = torch.cat([dec_x, torch.zeros((1, horizon, panel.shape[1]), device=device)], dim=1)

    ds = SkuPanelDataset(panel, seq_len, label_len, horizon)
    enc_mark = torch.tensor(ds._time_features(panel.shape[0] - seq_len, panel.shape[0]),
                            dtype=torch.float32).unsqueeze(0).to(device)
    dec_mark = torch.tensor(ds._time_features(panel.shape[0] - label_len, panel.shape[0] + horizon),
                            dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(enc_x, enc_mark, dec_x, dec_mark).squeeze(0).cpu().numpy()  # (horizon, N)

    rows = []
    for j, sku in enumerate(skus):
        for h, val in enumerate(out[:, j], start=1):
            # 2. CRITICAL: Inverse transform (expm1) to convert back to physical retail units
            physical_qty = np.expm1(val)
            rows.append((sku, h, max(0.0, float(physical_qty))))
            
    return pd.DataFrame(rows, columns=["StockCode", "Horizon", "Forecast"])


def train_models(df_long: pd.DataFrame, params: dict = None):
    """
    Orchestrates the training of one Non-Stationary Transformer PER seasonal cluster,
    matching the architectural logic of our Linear Regression and LightGBM models.
    """
    print("Training NS-Transformers per Seasonal Profile Cluster...")
    cluster_models = {}
    
    # Identify unique clusters strictly from the training period
    train_df = df_long[df_long['Week'] < TEST_CUTOFF_DT]
    unique_clusters = train_df['profile_cluster_id'].dropna().unique()

    for cluster_id in sorted(unique_clusters):
        # Isolate the SKUs belonging to this specific behavioral cluster
        cluster_skus = train_df[train_df['profile_cluster_id'] == cluster_id]['StockCode'].unique().tolist()
        
        if len(cluster_skus) < 2:
            continue # Skip clusters that don't have enough series for cross-attention
            
        print(f"\n--- Training NST for Cluster {int(cluster_id)} ({len(cluster_skus)} SKUs) ---")
        
        # Train a dedicated multivariate model for this specific shape group
        model, meta = train_ns_transformer(df_long, cluster_skus, params=params)
        
        # CRITICAL FIX: Attach the active PyTorch model object as 'live_model' 
        # so the predict_models function can use it immediately for evaluation.
        cluster_models[cluster_id] = {
            "model_state": model.state_dict(),
            "live_model": model, 
            "skus": cluster_skus,
            "meta": meta
        }
        
    return cluster_models


def predict_models(cluster_models: dict, test: pd.DataFrame, df_long: pd.DataFrame, horizon_weeks: int = 12):
    """
    Wraps the NST inference function to match the standard pipeline API.
    Maps the horizon predictions back to the chronological test DataFrame.
    Uses vectorized merge instead of row-by-row assignment for speed.
    """
    print("Predicting on Test Set using NS-Transformers...")
    test = test.copy()
    test['Actual_Qty'] = test['Quantity']  # Real physical quantities

    all_forecasts = []
    for cluster_id, artifact in cluster_models.items():
        live_model = artifact.get("live_model", None)
        if not live_model:
            continue

        cluster_skus = artifact["skus"]

        # Call the specialized Transformer prediction function
        fcst_df = predict_ns_transformer(live_model, df_long, cluster_skus, horizon=horizon_weeks)

        # Build a date-horizon mapping ONCE per cluster (vectorized)
        for sku in cluster_skus:
            sku_dates = test.loc[test['StockCode'] == sku, 'Week'].sort_values().head(horizon_weeks).tolist()
            sku_fcst = fcst_df[fcst_df['StockCode'] == sku].sort_values('Horizon')

            for h, date in enumerate(sku_dates):
                if h < len(sku_fcst):
                    all_forecasts.append({
                        'StockCode': sku,
                        'Week': date,
                        'Predicted_Qty': sku_fcst.iloc[h]['Forecast']
                    })

    # Vectorized merge instead of O(N*H*M) row-by-row assignment
    if all_forecasts:
        fcst_merge = pd.DataFrame(all_forecasts)
        test = test.merge(fcst_merge, on=['StockCode', 'Week'], how='left', suffixes=('_old', ''))
        # If there was already a Predicted_Qty column, merge fills the new one
        if 'Predicted_Qty_old' in test.columns:
            test['Predicted_Qty'] = test['Predicted_Qty'].fillna(test['Predicted_Qty_old'])
            test = test.drop(columns=['Predicted_Qty_old'])
    else:
        test['Predicted_Qty'] = np.nan

    print("Predictions Complete!")
    return test


def evaluate_models(test: pd.DataFrame):
    """
    Standardized evaluation. Since NST predict_ns_transformer already applies expm1 
    (inverse scaling), we don't need to do it here. We just compute the metrics.
    """
    print("\nEvaluating model (raw Quantity)...")
    
    # Group by profile_cluster_id and Date (Week)
    test['Cluster'] = test['profile_cluster_id']
    test['Date'] = test['Week']
    
    cluster_eval = test.dropna(subset=['Actual_Qty', 'Predicted_Qty'])[['Cluster', 'StockCode', 'Date', 'Actual_Qty', 'Predicted_Qty']].copy()

    summary = compute_cluster_metrics(cluster_eval)
    return cluster_eval, summary


def save_artifacts(cluster_models: dict, sku_clusters: dict, artifacts_dir: str = "../agent/artifacts"):
    """
    Saves the Transformer states and SKU mappings for the Agentic Layer.
    """
    print(f"Saving Cluster NS-Transformer artifacts to {artifacts_dir}...")
    os.makedirs(artifacts_dir, exist_ok=True)

    file_name = "nst_cluster_models.pkl"
    path = os.path.join(artifacts_dir, file_name)
    
    # We strip out the heavy 'live_model' object and only save the state_dict for PyTorch serialization
    safe_models = {}
    for cid, data in cluster_models.items():
        safe_models[cid] = {
            "model_state": data["model_state"],
            "skus": data["skus"],
            "meta": data["meta"]
        }
    
    artifact = {
        "cluster_models": safe_models,
        "sku_clusters": {k: v for k, v in sku_clusters.items()}
    }
    
    joblib.dump(artifact, path)
    print(f"Successfully saved {path}")


def run_nst_pipeline(file_path: str, plot: bool = False):
    """
    Complete pipeline to load data, train Transformers, predict, evaluate, and visualize results.
    """
    df_long = load_processed_data(file_path)
    
    # 1. We don't need X_train/X_test standard splits because NST uses a sliding window panel
    train_df = df_long[df_long['Week'] < TEST_CUTOFF_DT].copy()
    test_df  = df_long[df_long['Week'] >= TEST_CUTOFF_DT].copy()
    
    # 2. Train the models per cluster
    # Note: make sure train_models attaches the actual PyTorch model to the dictionary 
    # as "live_model" so we can run predictions immediately below.
    cluster_models = train_models(df_long)
    # 3. Predict & Evaluate
    test_df = predict_models(cluster_models, test_df, df_long, horizon_weeks=12)
    cluster_eval, summary = evaluate_models(test_df)
    
    # 4. Save Artifacts for the Agent
    sku_clusters = df_long.drop_duplicates(subset=['StockCode']).set_index('StockCode')['profile_cluster_id'].to_dict()
    # Path uses the same logic as your other files
    agent_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'agent', 'artifacts')
    save_artifacts(cluster_models, sku_clusters, artifacts_dir=agent_dir)

    if plot:
        plot_cluster_portfolio(cluster_eval, summary, model_label="NS-Transformer Forecast")
        analyze_time_periods(test_df)
    
    return cluster_models, test_df, cluster_eval, summary


if __name__ == "__main__":
    # Ensure correct project root path resolution
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed_retail_data.parquet")
    
    _, _, _, summary = run_nst_pipeline(DATA_PATH, plot=False)
    print("\n=== Non-Stationary Transformer Evaluation Summary ===")
    print(summary.to_markdown())