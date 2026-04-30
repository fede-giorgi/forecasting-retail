"""
FastAPI server exposing the forecast lookup as an HTTP tool for n8n.

n8n flow:
  Webhook (chat input)
    → AI Agent node with system prompt "You are a forecast assistant..."
    → HTTP Request tool: POST {SERVER}/forecast {"prompt": "<user msg>"}
    → return the response text to the user

Run locally:
    uvicorn agent.server:app --host 0.0.0.0 --port 8000

Persistence: the server expects pre-computed parquet artifacts written by the
playground (`data/processed/agent_summary.parquet`, `agent_horizon.parquet`).
Hot-reload them by hitting POST /reload.
"""
from pathlib import Path
import os

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .forecast_service import get_forecast_for_prompt

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
SUMMARY_PATH = ROOT / "data" / "processed" / "agent_summary.parquet"
HORIZON_PATH = ROOT / "data" / "processed" / "agent_horizon.parquet"

app = FastAPI(title="Retail Forecast Agent", version="0.1.0")
_state: dict = {"summary": None, "horizon": None}


class ForecastRequest(BaseModel):
    prompt: str


class ForecastResponse(BaseModel):
    text: str


def _load_artifacts():
    if not SUMMARY_PATH.exists() or not HORIZON_PATH.exists():
        raise FileNotFoundError(
            f"Run the playground first to generate {SUMMARY_PATH} and {HORIZON_PATH}."
        )
    _state["summary"] = pd.read_parquet(SUMMARY_PATH)
    _state["horizon"] = pd.read_parquet(HORIZON_PATH)


@app.on_event("startup")
def _startup():
    try:
        _load_artifacts()
    except FileNotFoundError as e:
        # boot anyway so /reload can fix it later
        print(f"[startup] {e}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "artifacts_loaded": _state["summary"] is not None,
        "n_skus": int(_state["summary"]["StockCode"].nunique()) if _state["summary"] is not None else 0,
    }


@app.post("/reload")
def reload_artifacts():
    _load_artifacts()
    return {"status": "reloaded", "n_skus": int(_state["summary"]["StockCode"].nunique())}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    if _state["summary"] is None:
        raise HTTPException(503, detail="Artifacts not loaded — POST /reload after playground writes them.")
    text = get_forecast_for_prompt(req.prompt, _state["summary"], _state["horizon"])
    return ForecastResponse(text=text)
