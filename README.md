# Forecasting Retail

Weekly SKU-level demand forecasting on the UCI *Online Retail II* dataset (UK gift-ware retailer, 2009-2011). Per-cluster training (Linear/Ridge, Prophet, LightGBM-Tweedie) on a HDBSCAN segmentation built from Gemini description embeddings + demand & commercial profiles. A LangChain ReAct agent in the terminal exposes the forecasts in natural language.

## Layout

```
forecasting-retail/
├── data/
│   ├── raw/                 online_retail_II.xlsx (committed)
│   └── processed/           parquet artifacts (gitignored — embeddings, agent tables)
├── docs/
│   ├── strategy_roadmap.ipynb   rationale + status badges + pseudocode
│   ├── reference/               prior team's PDFs
│   └── archive/                 prior team's monolith notebook
├── notebooks/
│   ├── playground.ipynb         central orchestrator (top-to-bottom)
│   ├── eda_playground.ipynb     dataset exploration
│   └── *_playground.ipynb       per-model sandboxes
├── src/
│   ├── tools/                   load, clean, feature_engineering, embeddings,
│   │                            clustering, evaluation, visualization
│   └── models/                  naive, linear_regression, prophet_model,
│                                lightgbm_recursive, sarimax, deepar,
│                                ns_transformer, selection
├── agent/                       LangChain ReAct agent (terminal, rich-rendered)
│   ├── chatbot.py               main entry: python -m agent.chatbot
│   ├── inference/predict.py     model lookup + forecast generation
│   └── artifacts/               trained pickles written by the playground
├── scripts/                     one-off processing scripts
├── requirements.txt
└── .env.example                 copy to .env and fill in keys
```

## Setup

```bash
# Python 3.14 (recommended). 3.12 also works.
brew install python@3.14 libomp
cd forecasting-retail
python3.14 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt

cp .env.example .env            # then fill in the keys you need
```

### What to put in `.env`

At minimum: an LLM key matching your `LLM_PROVIDER` (default OpenAI), and `GEMINI_API_KEY` if you ever need to (re)build embeddings — the embeddings cache means only the first playground run hits the API.

| Variable | Required? | Notes |
|---|---|---|
| `LLM_PROVIDER` | optional | `openai` (default) / `gemini` / `claude` / `ollama` |
| `OPENAI_API_KEY` | if provider=openai | gpt-4o-mini by default |
| `GOOGLE_API_KEY` | if provider=gemini | LangChain reads this name |
| `ANTHROPIC_API_KEY` | if provider=claude | |
| `GEMINI_API_KEY` | for embeddings | Same key as `GOOGLE_API_KEY` works |
| `AWS_*` | optional | Only for DeepAR on SageMaker |

## Run

### 1. Run the playground (produces processed data + trained model artifacts)

```bash
jupyter lab notebooks/playground.ipynb
```
Run top-to-bottom. Stages: load → clean → features → embedding → clustering → per-cluster model training → evaluation → write artifacts.

This produces:
- `data/processed/processed_retail_data.parquet` — the full panel the chatbot reads on startup
- `agent/artifacts/{lgb,prophet,lr}_cluster_models.pkl` — trained per-cluster models

### 2. Talk to the agent

```bash
python -m agent.chatbot
```

You'll get a rich-rendered terminal chat. Try:
- `Tell me about product 22423`
- `Forecast 85123A for the next 8 weeks`
- `Compare LightGBM vs Prophet on 22423 over 12 weeks`
- `What's the seasonal profile of 47566?`

Type `exit` / `quit` / `q` (or Ctrl+C) to leave.

The agent has two tools: `run_forecast(stock_code, model, horizon_weeks)` and `get_product_info(stock_code)`. The system prompt nudges it to call `get_product_info` for profiling questions and to compare two models when forecasting.

## Pipeline

1. Load both Excel sheets (2009-2010 + 2010-2011)
2. Split sales (target) from returns (feature only); detail in `docs/strategy_roadmap.ipynb` §2
3. Build temporal + return-rate + lag/rolling + price + demand-class + commercial-profile features (`src/tools/feature_engineering.py`)
4. Embed canonical SKU descriptions with Gemini Embedding 2 (parquet-cached)
5. HDBSCAN cluster on `[UMAP(emb) ⊕ demand profile ⊕ commercial profile]` → `profile_cluster_id`
6. **One model per cluster** (Linear/Ridge, Prophet, LightGBM-Tweedie); trained on cluster-aggregated demand, predictions broadcast back to constituent SKUs
7. Evaluate with **WMAPE as headline** + Median MAPE + MAE per cluster (see `src/tools/evaluation.py`)
8. Pickle each cluster's trained model into `agent/artifacts/` for the LangChain agent to load

### Current performance

LightGBM is the best per-cluster model (WMAPE ~50%), followed by Prophet and Ridge (~60%), with Naive as the floor. Why MAPE looks high: at ~10 units/week per SKU, a 4-unit miss is already 40-50% MAPE no matter how good the model is. **Read WMAPE first** — it pools by volume, so high-volume SKUs (the ones that drive inventory cost) dominate the score.

## Demand classification (Syntetos-Boylan)

`feature_engineering.py` tags every SKU with one of `smooth / erratic / intermittent / lumpy` based on ADI (avg weeks between sales) and CV² (squared coefficient of variation). The thresholds (1.32 / 0.49) come from Syntetos & Boylan 2005 — they're the analytical break-even points where Croston's method beats simple smoothing, and where SBA beats Croston. See the comment block in `calculate_demand_profile()` for the full derivation.

## License

MIT — see `LICENSE`.
