"""
HDBSCAN clustering on [Gemini embedding ⊕ demand profile ⊕ commercial profile].

Pipeline:
  1) reduce 768-d embeddings via UMAP to ~32 dims (cosine metric)
  2) standard-scale demand + commercial profiles independently
  3) concat → HDBSCAN with min_cluster_size=20 (Euclidean on the reduced space)

Outliers are labeled `-1` and represent the long-tail of rare/peculiar SKUs.
We do NOT fit a downstream model on `-1` — we either fall back to baseline
(seasonal-naive) or train a single "noise" cluster model. The cluster_id is
used as a static_cat feature in DeepAR / NS-Transformer.

The dependencies are heavy (umap-learn, hdbscan). Both are imported lazily.
"""
from typing import Iterable
import numpy as np
import pandas as pd

from .embeddings import embeddings_as_matrix


def cluster_skus(
    emb_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    commercial_df: pd.DataFrame,
    umap_dim: int = 32,
    min_cluster_size: int = 20,
    demand_cols: Iterable[str] | None = None,
    commercial_cols: Iterable[str] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Returns DataFrame [StockCode, cluster_id]. Aligns rows on StockCode across the three inputs."""
    import umap
    import hdbscan
    from sklearn.preprocessing import StandardScaler

    demand_cols = list(demand_cols) if demand_cols else ["ADI", "CV2", "share_zero_weeks"]
    commercial_cols = list(commercial_cols) if commercial_cols else [
        "price_median", "mean_basket_size", "n_unique_customers", "country_uk_share",
    ]

    # 1) inner-join all three on StockCode to keep only SKUs present everywhere
    merged = (
        emb_df[["StockCode", "embedding"]]
        .merge(demand_df[["StockCode"] + demand_cols], on="StockCode")
        .merge(commercial_df[["StockCode"] + commercial_cols], on="StockCode")
    )

    skus, embeddings = embeddings_as_matrix(merged.rename(columns={"embedding": "embedding"}))

    # 2) UMAP reduction on embeddings (cosine on unit vectors → Euclidean ≈ cosine)
    reducer = umap.UMAP(n_components=umap_dim, metric="cosine", random_state=random_state)
    emb_reduced = reducer.fit_transform(embeddings)

    # 3) scale demand + commercial features independently then concat
    demand_scaled = StandardScaler().fit_transform(merged[demand_cols].fillna(0).values)
    comm_scaled = StandardScaler().fit_transform(merged[commercial_cols].fillna(0).values)

    feat = np.concatenate([emb_reduced, demand_scaled, comm_scaled], axis=1)

    # 4) HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    labels = clusterer.fit_predict(feat)

    return pd.DataFrame({"StockCode": skus, "cluster_id": labels})


def cluster_summary(clusters: pd.DataFrame) -> pd.DataFrame:
    return (
        clusters.groupby("cluster_id")
        .size().reset_index(name="n_skus")
        .sort_values("n_skus", ascending=False)
    )
