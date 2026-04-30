"""
Gemini Embedding 2 helper for SKU descriptions.

API surface mirrors Google's quickstart: a single
`client.models.embed_content(model=..., contents=[...], config=...)` call.
We only deal with text here (no images / audio) since the dataset has only
SKU description strings — multimodal Parts would be wasted bandwidth.

Decisions tuned for this project:
- task_type="CLUSTERING" by default: our primary use is HDBSCAN on SKUs, where
  the model produces vectors with cosine geometry optimized for clustering. If
  embeddings are fed directly as static_real features into DeepAR / iTransformer,
  switch to "RETRIEVAL_DOCUMENT".
- output_dimensionality=768: MRL sweet spot per Google. 3072 is overkill for
  ~5k SKUs and 4× the storage; 1536 brings no measurable clustering quality
  improvement on short product strings.
- L2-normalize by default: cosine distance == 1 - dot for unit vectors, which is
  what HDBSCAN / UMAP expect. Skip normalization only if you intend to use
  Euclidean directly on the raw vector.

One canonical embedding per StockCode, cached to parquet so we don't re-pay the
API cost on subsequent runs (the budget constraint matters).
"""
from pathlib import Path
import os
import re
import time

import numpy as np
import pandas as pd

_RETRY_DELAY_RE = re.compile(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'")


def _suggested_delay(exc: Exception) -> float | None:
    """Pick the `retryDelay` Gemini suggests on 429 errors (returns seconds)."""
    m = _RETRY_DELAY_RE.search(str(exc))
    return float(m.group(1)) if m else None

MODEL = "gemini-embedding-2-preview"
DEFAULT_DIM = 768
# Gemini Embedding 2 is multimodal: a list passed to `contents=` is interpreted
# as MULTIPLE PARTS of ONE multimodal item, not N separate items. To get N
# embeddings we wrap each text in its own Content (see embed_texts).
DEFAULT_BATCH = 50              # preview limits are stricter than text-embedding-004
DEFAULT_TASK = "CLUSTERING"


def canonical_descriptions(sales: pd.DataFrame) -> pd.DataFrame:
    """One row per StockCode, picking the most frequent description.
    Tiebreak on the longest unique string (avoids 'CHRISTMAS GIFT' beating
    'WHITE METAL CHRISTMAS GIFT BOX' just by ordering)."""
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
        out["desc_canonical"]
        .astype(str).str.strip().str.upper()
        .str.replace(r"\s+", " ", regex=True)
    )
    return out


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (v / norms).astype(np.float32)


def embed_texts(
    texts: list[str],
    dim: int = DEFAULT_DIM,
    batch_size: int = DEFAULT_BATCH,
    task_type: str = DEFAULT_TASK,
    normalize: bool = True,
    client=None,
) -> np.ndarray:
    """Call gemini-embedding-2-preview with batching + retry. Returns (N, dim)."""
    from google import genai
    from google.genai import types

    if client is None:
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        # Wrap each string in its own Content so the API returns N embeddings,
        # not 1 (see DEFAULT_BATCH note above).
        contents = [types.Content(parts=[types.Part.from_text(text=t)]) for t in chunk]
        for attempt in range(6):
            try:
                resp = client.models.embed_content(
                    model=MODEL,
                    contents=contents,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=dim,
                    ),
                )
                got = list(resp.embeddings)
                if len(got) != len(chunk):
                    raise RuntimeError(
                        f"API returned {len(got)} embeddings for batch of {len(chunk)}"
                    )
                out.extend(e.values for e in got)
                break
            except Exception as exc:
                if attempt == 5:
                    raise
                # Respect Gemini's `retryDelay` (e.g. '28s' on 429) when present;
                # otherwise fall back to exponential backoff.
                wait = _suggested_delay(exc)
                wait = (wait + 1.0) if wait is not None else min(60.0, 2 ** attempt)
                print(f"[embed_texts] retry {attempt + 1} in {wait:.0f}s: {str(exc)[:80]}")
                time.sleep(wait)

    arr = np.asarray(out, dtype=np.float32)
    return _l2_normalize(arr) if normalize else arr


def embed_sku_descriptions(
    sales: pd.DataFrame,
    cache_path: str | Path | None = None,
    dim: int = DEFAULT_DIM,
    batch_size: int = DEFAULT_BATCH,
    task_type: str = DEFAULT_TASK,
    normalize: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """Returns DataFrame [StockCode, desc_canonical, embedding].
    Cached to parquet — first run hits the API, subsequent runs are free."""
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists() and not force:
        return pd.read_parquet(cache_path)

    canonical = canonical_descriptions(sales)
    vecs = embed_texts(
        canonical["desc_canonical"].tolist(),
        dim=dim, batch_size=batch_size, task_type=task_type, normalize=normalize,
    )
    canonical["embedding"] = list(vecs)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        canonical.to_parquet(cache_path, index=False)
    return canonical


def embeddings_as_matrix(emb_df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Split into (sku_list, matrix) for clustering / dim-reduction."""
    skus = emb_df["StockCode"].astype(str).tolist()
    matrix = np.vstack(emb_df["embedding"].to_list()).astype(np.float32)
    return skus, matrix
