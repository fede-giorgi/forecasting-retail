# Forecasting Retail

Weekly SKU-level demand forecasting on the UCI *Online Retail II* dataset (UK gift-ware retailer, 2009-2011). Per-SKU model selection (Naive, SARIMAX-seasonal, Prophet, LightGBM-Tweedie) on rolling-origin folds, plus global models (DeepAR, Non-Stationary Transformer) and HDBSCAN clustering with Gemini description embeddings. Exposes a natural-language forecast tool for an n8n agent.

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
│   └── playground.ipynb         central orchestrator (the only notebook to run)
├── src/
│   ├── tools/                   load, clean, features, embeddings, clustering, evaluation
│   └── modelling/               naive, sarimax, prophet, lightgbm, deepar, ns_transformer
├── agent/                       n8n-facing service (FastAPI + prompt parser)
└── infra/                       Docker Compose for n8n + forecast-api
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add GEMINI_API_KEY for embeddings
```

## Run

1. **Open the playground**: `jupyter lab notebooks/playground.ipynb` and run top-to-bottom. Stages: load → clean → features → embedding → clustering → model selection → forecast → agent artifacts.
2. **Boot the agent**: `cd infra && docker compose up -d --build` — see `infra/README.md`.

## Pipeline

1. Load both Excel sheets (2009-2010 + 2010-2011)
2. Split sales (target) from returns (feature only); detail in `docs/strategy_roadmap.ipynb` §2
3. Build weekly + return-rate + calendar + lag/rolling + price + demand-class + commercial-profile features
4. Embed canonical SKU descriptions with Gemini Embedding 2 (parquet-cached)
5. HDBSCAN cluster on `[UMAP(emb) ⊕ demand profile ⊕ commercial profile]`
6. Per-SKU model selection on rolling-origin (3 folds × 12 weeks)
7. Final 12-week forecast + revenue translation
8. Agent artifacts to `data/processed/` consumed by the FastAPI server

## License

MIT — see `LICENSE`.
