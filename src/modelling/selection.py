"""
Per-SKU model selection with rolling-origin CV.

Strategy:
  - For each SKU, generate N rolling folds (default 3 × 12-week test).
  - Fold 1 is "validation": score every model in the registry on it.
  - Pick the model with lowest mean Block-APE on fold 1.
  - Score the chosen model on folds 2..N → these are the held-out test estimates.

Why rolling-origin (vs single train/val/test): with ~106 weekly observations
a single split is high-variance. 3 folds give us 3 independent test windows
with ~12 weeks each, surfacing recency-dependent regressions.

Output:
  choices            DataFrame[StockCode, Chosen_Model, Val_Block_MAPE_<m>...]
  test_block         long-form per-fold per-SKU block APE for the chosen model
  validation_block   long-form per-SKU block APE per model on fold 1
"""
from typing import Callable
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..tools.evaluation import rolling_block_evaluate, block_summary, rolling_origin_folds


def select_best_model(
    weekly_sku: pd.DataFrame,
    skus: list[str],
    model_registry: dict[str, Callable],
    n_folds: int = 3,
    test_size: int = 12,
    block_size: int = 4,
    min_train: int = 70,
    series_builder: Callable[[pd.DataFrame, str], pd.Series] | None = None,
) -> dict[str, pd.DataFrame]:
    if series_builder is None:
        from ..tools.features import build_series_for_sku
        series_builder = build_series_for_sku

    sel_rows, val_blocks, test_blocks = [], [], []

    for sku in tqdm(skus, desc="Model selection"):
        s = series_builder(weekly_sku, sku)
        folds = rolling_origin_folds(s, n_folds=n_folds, test_size=test_size, min_train=min_train)
        if len(folds) < 2:
            continue  # need at least 1 val + 1 test fold

        val_train, val_test = folds[0]
        test_folds = folds[1:]

        # 1) score every model on the validation fold
        val_scores = {}
        for name, mf in model_registry.items():
            ve = rolling_block_evaluate(val_train, val_test, mf, block_size=block_size)
            ve["StockCode"] = sku; ve["Model"] = name
            vb = block_summary(ve, ["StockCode", "Model"])
            val_blocks.append(vb)
            val_scores[name] = float(vb["Block_APE"].mean())

        best = min(val_scores, key=val_scores.get)
        row = {
            "StockCode": sku,
            "N_Folds_Used": len(folds),
            "Val_Train_Weeks": len(val_train),
            "Chosen_Model": best,
            "Best_Val_Block_MAPE": val_scores[best],
        }
        for name in model_registry:
            row[f"Val_Block_MAPE_{name}"] = val_scores.get(name, np.nan)
        sel_rows.append(row)

        # 2) score the chosen model on the remaining folds (true held-out)
        for fold_id, (tr, te) in enumerate(test_folds, start=2):
            te_eval = rolling_block_evaluate(tr, te, model_registry[best], block_size=block_size)
            te_eval["StockCode"] = sku
            te_eval["Chosen_Model"] = best
            tb = block_summary(te_eval, ["StockCode", "Chosen_Model"])
            tb["Fold"] = fold_id
            test_blocks.append(tb)

    return {
        "choices": pd.DataFrame(sel_rows),
        "validation_block": pd.concat(val_blocks, ignore_index=True) if val_blocks else pd.DataFrame(),
        "test_block": pd.concat(test_blocks, ignore_index=True) if test_blocks else pd.DataFrame(),
    }
