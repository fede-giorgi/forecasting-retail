"""
Gemini Embedding 2 helper for SKU descriptions.

One canonical embedding per StockCode, cached to parquet so we don't
re-pay the API cost between runs.
"""
from pathlib import Path
import os
import time

import numpy as np
import pandas as pd

MODEL = "gemini-embedding-2-preview"
DEFAULT_DIM = 768            # MRL: 3072 / 1536 / 768 recommended; 768 is the cost/quality sweet spot
DEFAULT_BATCH = 100          # Gemini API accepts up to ~250 contents per call; 100 is conservative
DEFAULT_TASK = "CLUSTERING"  # use "RETRIEVAL_DOCUMENT" if embeddings feed downstream models as features


def canonical_descriptions(sales: pd.DataFrame) -> pd.DataFrame:
    """One row per StockCode with the most frequent (longest as tiebreak) description."""
    def pick(s: pd.Series) -> str:
        modes = s.mode()
        if len(modes) == 1:
            return modes.iat[0]
        return max(s.dropna().unique(), key=len)

    out = (
        sales.groupby("StockCode")["Description"]
        .agg(pick)
        .reset_index()
        .rename(columns={"Description": "desc_canonical"})
    )
    out["desc_canonical"] = (
        out["desc_canonical"].astype(str).str.strip().str.upper().str.replace(r"\s+", " ", regex=True)
    )
    return out


def embed_texts(
    texts: list[str],
    dim: int = DEFAULT_DIM,
    batch_size: int = DEFAULT_BATCH,
    task_type: str = DEFAULT_TASK,
    client=None,
) -> np.ndarray:
    """Embed a list of strings; returns ndarray of shape (len(texts), dim)."""
    from google import genai
    from google.genai import types

    if client is None:
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        for attempt in range(3):
            try:
                resp = client.models.embed_content(
                    model=MODEL,
                    contents=chunk,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=dim,
                    ),
                )
                vectors.extend(e.values for e in resp.embeddings)
                break
            except Exception as exc:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
                print(f"[embed_texts] retry {attempt + 1} after error: {exc}")
    return np.asarray(vectors, dtype=np.float32)


def embed_sku_descriptions(
    sales: pd.DataFrame,
    cache_path: str | Path | None = None,
    dim: int = DEFAULT_DIM,
    batch_size: int = DEFAULT_BATCH,
    task_type: str = DEFAULT_TASK,
    force: bool = False,
) -> pd.DataFrame:
    """
    Embed each SKU's canonical description with Gemini Embedding 2.

    Returns a DataFrame with columns [StockCode, desc_canonical, embedding]
    where `embedding` is a list[float] of length `dim`.

    If `cache_path` is provided, embeddings are read from / written to parquet
    so subsequent runs don't re-call the API.
    """
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists() and not force:
        return pd.read_parquet(cache_path)

    canonical = canonical_descriptions(sales)
    vectors = embed_texts(
        canonical["desc_canonical"].tolist(),
        dim=dim,
        batch_size=batch_size,
        task_type=task_type,
    )
    canonical["embedding"] = list(vectors)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        canonical.to_parquet(cache_path, index=False)

    return canonical


def embeddings_as_matrix(emb_df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Convenience: split an embeddings DataFrame into (sku_list, matrix) for clustering."""
    skus = emb_df["StockCode"].astype(str).tolist()
    matrix = np.vstack(emb_df["embedding"].to_list()).astype(np.float32)
    return skus, matrix
