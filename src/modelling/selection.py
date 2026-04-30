from typing import Callable
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..tools.splits import split_train_val_test
from ..tools.evaluation import rolling_block_evaluate, block_summary


def select_best_model(
    weekly_sku: pd.DataFrame,
    skus: list[str],
    model_registry: dict[str, Callable],
    block_size: int = 4,
    val_size: int = 12,
    test_size: int = 12,
    series_builder: Callable[[pd.DataFrame, str], pd.Series] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    For each SKU:
      1) split train/val/test
      2) score each model on validation via rolling-block MAPE
      3) pick the lowest, evaluate it on test using train+val as history

    Returns dict with: choices, validation_block, test_block.
    """
    if series_builder is None:
        from ..tools.features import build_series_for_sku
        series_builder = build_series_for_sku

    selection_rows, val_blocks, test_blocks = [], [], []

    for sku in tqdm(skus, desc="Model selection"):
        s = series_builder(weekly_sku, sku)
        split = split_train_val_test(s, val_size=val_size, test_size=test_size)
        if split is None:
            continue
        train, val, test = split

        val_scores = {}
        for name, mf in model_registry.items():
            val_eval = rolling_block_evaluate(train, val, mf, block_size=block_size)
            val_eval["StockCode"] = sku
            val_eval["Model"] = name
            vb = block_summary(val_eval, ["StockCode", "Model"])
            val_blocks.append(vb)
            val_scores[name] = float(vb["Block_APE"].mean())

        best = min(val_scores, key=val_scores.get)
        row = {
            "StockCode": sku,
            "Train_Weeks": len(train),
            "Val_Weeks": len(val),
            "Test_Weeks": len(test),
            "Chosen_Model": best,
            "Best_Val_Block_MAPE": val_scores[best],
        }
        for name in model_registry:
            row[f"Val_Block_MAPE_{name}"] = val_scores.get(name, np.nan)
        selection_rows.append(row)

        history_tv = pd.concat([train, val])
        test_eval = rolling_block_evaluate(history_tv, test, model_registry[best], block_size=block_size)
        test_eval["StockCode"] = sku
        test_eval["Chosen_Model"] = best
        tb = block_summary(test_eval, ["StockCode", "Chosen_Model"])
        test_blocks.append(tb)

    return {
        "choices": pd.DataFrame(selection_rows),
        "validation_block": pd.concat(val_blocks, ignore_index=True) if val_blocks else pd.DataFrame(),
        "test_block": pd.concat(test_blocks, ignore_index=True) if test_blocks else pd.DataFrame(),
    }
