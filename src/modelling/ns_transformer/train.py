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
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .architecture import NonStationaryTransformer

DEFAULT = {
    "SEQ_LEN": 52, "LABEL_LEN": 12, "PRED_LEN": 12,
    "D_MODEL": 128, "N_HEADS": 4, "E_LAYERS": 2, "D_LAYERS": 1, "D_FF": 256,
    "DROPOUT": 0.1, "BATCH_SIZE": 32, "LR": 1e-4, "EPOCHS": 30, "PATIENCE": 5,
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
        return np.stack([wk_of_year, sin_year], axis=-1).astype(np.float32)

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

    panel = _build_panel(weekly_sku, skus)         # (T, N_skus)
    val_size = max(int(panel.shape[0] * 0.15), p["SEQ_LEN"] + p["PRED_LEN"] + 1)
    train_data = panel[:-val_size]
    val_data = panel[-val_size:]

    train_ds = SkuPanelDataset(train_data, p["SEQ_LEN"], p["LABEL_LEN"], p["PRED_LEN"])
    val_ds = SkuPanelDataset(val_data, p["SEQ_LEN"], p["LABEL_LEN"], p["PRED_LEN"])
    train_loader = DataLoader(train_ds, batch_size=p["BATCH_SIZE"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=p["BATCH_SIZE"], shuffle=False)

    model = NonStationaryTransformer(
        enc_in=panel.shape[1], c_out=panel.shape[1],
        seq_len=p["SEQ_LEN"], label_len=p["LABEL_LEN"], pred_len=p["PRED_LEN"],
        d_model=p["D_MODEL"], n_heads=p["N_HEADS"],
        e_layers=p["E_LAYERS"], d_layers=p["D_LAYERS"],
        d_ff=p["D_FF"], dropout=p["DROPOUT"],
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=p["LR"])
    crit = nn.HuberLoss()  # Huber ≈ MAE on log1p — better aligned with WMAPE than MSE

    best, patience, best_state = float("inf"), 0, None
    for epoch in range(p["EPOCHS"]):
        model.train()
        for x, y, xm, ym in train_loader:
            x, y, xm, ym = x.to(device), y.to(device), xm.to(device), ym.to(device)
            yhat = model(x, xm, y, ym)
            loss = crit(yhat, y[:, -p["PRED_LEN"] :, :])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        v = []
        with torch.no_grad():
            for x, y, xm, ym in val_loader:
                x, y, xm, ym = x.to(device), y.to(device), xm.to(device), ym.to(device)
                v.append(crit(model(x, xm, y, ym), y[:, -p["PRED_LEN"] :, :]).item())
        avg = float(np.mean(v)) if v else float("inf")
        print(f"epoch {epoch + 1:>2}  val={avg:.4f}")

        if avg < best:
            best, patience = avg, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= p["PATIENCE"]:
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
    seq_len: int = 52,
    label_len: int = 12,
    device: torch.device | None = None,
) -> pd.DataFrame:
    device = device or get_device()
    model.eval()
    panel = _build_panel(weekly_sku, skus)
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
            rows.append((sku, h, max(0.0, float(val))))
    return pd.DataFrame(rows, columns=["StockCode", "Horizon", "Forecast"])
