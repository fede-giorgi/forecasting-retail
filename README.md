# Forecasting Retail

Weekly SKU-level demand forecasting on the UCI *Online Retail II* dataset. Compares baseline, SARIMAX, and LightGBM models per SKU and produces 12-week forecasts plus revenue translation.

## Layout

```
Datasets/   raw input (online_retail_II.xlsx)
PDF/        deliverable write-ups & final presentation
src/tools/  reusable loading + cleaning utilities
scripts/    exploratory notebooks (playground.ipynb)
TrendSetters_Retail_Model_Selection_Deliverable_2.ipynb   main pipeline
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then open `TrendSetters_Retail_Model_Selection_Deliverable_2.ipynb` in Jupyter / VS Code.

## Pipeline

1. Load + clean raw transactions (drop cancellations, negatives)
2. Aggregate to weekly SKU level (Monday-aligned)
3. Train/validation/test split per SKU
4. Model selection on validation MAPE (Naive / SARIMAX / LightGBM)
5. Held-out test MAPE + 12-week forecasts
6. Demand → revenue via median SKU price

## License

MIT — see `LICENSE`.
