"""
Gemini Embedding 2 helper for SKU descriptions.

This module is responsible for converting textual product descriptions into 
mathematical vectors (embeddings) using Google's Gemini AI. This allows 
downstream machine learning models (like clustering algorithms or neural networks) 
to understand the semantic meaning of products (e.g., knowing that a 'RED MUG' 
is similar to a 'BLUE MUG').

Key Features:
- Task Type: Optimized for "CLUSTERING" by default, producing vectors designed 
  for distance-based algorithms like HDBSCAN.
- Dimensionality: Defaults to 768 dimensions, which provides the best balance 
  between accuracy and storage space for short product descriptions.
- Normalization: Outputs are L2-normalized to allow the use of simple Euclidean 
  distance in downstream clustering, mimicking Cosine similarity.
- Caching: Embeddings are saved locally (parquet) to avoid repaying API costs.
"""

from pathlib import Path
import os
import re
import time
import numpy as np
import pandas as pd
from google import genai
from google.genai import types

# Regular expression to extract the suggested wait time from Gemini API 429 Error (Rate Limit)
_RETRY_DELAY_REGEX = re.compile(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'")

# Core Gemini Configuration Constants
MODEL = "gemini-embedding-2-preview"
DEFAULT_DIM = 768

# API constraints: We process descriptions in batches to avoid overwhelming the server.
DEFAULT_BATCH = 50              
DEFAULT_TASK = "CLUSTERING"


def _suggested_delay(exception: Exception) -> float | None:
    """
    Parses the API exception message to find the exact 'retryDelay' 
    suggested by Google when hitting rate limits.
    """
    match = _RETRY_DELAY_REGEX.search(str(exception))
    return float(match.group(1)) if match else None


def canonical_descriptions(sales: pd.DataFrame) -> pd.DataFrame:
    """
    Finds the definitive (canonical) description for each unique product (StockCode).
    
    Since the same product might be typed slightly differently in different transactions,
    this function selects the most frequently used description. In case of a tie, 
    it picks the longest description (which is usually the most detailed one).

    Args:
        sales (pd.DataFrame): The raw sales data containing 'StockCode' and 'Description'.

    Returns:
        pd.DataFrame: A mapping table with columns ['StockCode', 'desc_canonical'].
    """
    def pick_best_description(descriptions_series: pd.Series) -> str:
        most_frequent = descriptions_series.mode()
        # If there is a clear winner, use it
        if len(most_frequent) == 1:
            return most_frequent.iat[0]
        # In case of a tie, select the longest unique string to preserve maximum detail
        unique_descriptions = descriptions_series.dropna().unique()
        return max(unique_descriptions, key=len)

    # Group by product code and apply our selection logic
    canonical_df = (
        sales.groupby("StockCode")["Description"]
        .agg(pick_best_description)
        .reset_index()
        .rename(columns={"Description": "desc_canonical"})
    )
    
    # Standardize the text: convert to string, strip whitespace, uppercase, and remove double spaces
    canonical_df["desc_canonical"] = (
        canonical_df["desc_canonical"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
    )
    
    return canonical_df


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """
    Mathematically normalizes a batch of vectors so they all have a length (magnitude) of 1.
    This ensures that distance calculations only capture direction (semantic meaning), not scale.
    """
    vector_magnitudes = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Prevent division by zero for completely empty vectors
    safe_magnitudes = np.where(vector_magnitudes == 0, 1.0, vector_magnitudes)
    return (vectors / safe_magnitudes).astype(np.float32)


def embed_texts(
    texts: list[str],
    dim: int = DEFAULT_DIM,
    batch_size: int = DEFAULT_BATCH,
    task_type: str = DEFAULT_TASK,
    normalize: bool = True,
    client=None,
    ) -> np.ndarray:
    """
    Calls the Google Gemini API to convert a list of strings into mathematical embeddings.
    Handles rate limits, batching, and exponential backoff retries automatically.
    
    Returns a 2D NumPy array of shape (Number of texts, Dimensions).
    """

    if client is None:
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    all_embeddings: list[list[float]] = []
    
    # Process the texts in chunks to respect API limits
    for index in range(0, len(texts), batch_size):
        text_batch = texts[index : index + batch_size]
        
        # Gemini API requires each string to be wrapped in a specific 'Content' object
        api_contents = [types.Content(parts=[types.Part.from_text(text=text)]) for text in text_batch]
        
        # We attempt the API call up to 6 times in case of temporary network or rate-limit issues
        for attempt in range(6):
            try:
                response = client.models.embed_content(
                    model=MODEL,
                    contents=api_contents,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=dim,
                    ),
                )
                
                returned_embeddings = list(response.embeddings)
                
                # Sanity check to ensure Google returned exactly one vector per input text
                if len(returned_embeddings) != len(text_batch):
                    raise RuntimeError(
                        f"API Error: Requested {len(text_batch)} embeddings, but received {len(returned_embeddings)}."
                    )
                    
                all_embeddings.extend(embedding.values for embedding in returned_embeddings)
                break  # Success! Break out of the retry loop.
                
            except Exception as exception:
                if attempt == 5:
                    raise  # If we fail 6 times, bubble up the error to stop the program
                
                # If we hit a rate limit, the API tells us exactly how long to wait.
                # Otherwise, we use exponential backoff (1s, 2s, 4s, 8s, 16s...)
                wait_time = _suggested_delay(exception)
                wait_time = (wait_time + 1.0) if wait_time is not None else min(60.0, 2 ** attempt)
                
                print(f"[Embedding Engine] Retrying chunk {index} (Attempt {attempt + 1}) in {wait_time:.0f} seconds. Error: {str(exception)[:80]}")
                time.sleep(wait_time)

    # Convert the raw lists of floats into an optimized NumPy matrix
    embeddings_matrix = np.asarray(all_embeddings, dtype=np.float32)
    
    if normalize:
        return _l2_normalize(embeddings_matrix)
    return embeddings_matrix


def embed_sku_descriptions(
    sales: pd.DataFrame,
    cache_path: str | Path | None = None,
    dim: int = DEFAULT_DIM,
    batch_size: int = DEFAULT_BATCH,
    task_type: str = DEFAULT_TASK,
    normalize: bool = True,
    force: bool = False,
    ) -> pd.DataFrame:
    """
    Main entry point for product embedding.
    Finds the canonical description for each product and calculates its embedding.
    
    Results are saved (cached) to a local Parquet file. On subsequent runs, it will 
    load the cached file instantly instead of making paid requests to the Google API.
    
    Returns a DataFrame with columns: [StockCode, desc_canonical, embedding].
    """
    if cache_path:
        cache_path = Path(cache_path)
        # If cache exists and we are not forcing a recalculation, load and return it instantly
        if cache_path.exists() and not force:
            print(f"Loading cached embeddings from {cache_path}...")
            return pd.read_parquet(cache_path)

    print("Extracting canonical descriptions...")
    canonical_df = canonical_descriptions(sales)
    
    print("Sending descriptions to Gemini API for embedding...")
    embeddings_matrix = embed_texts(
        canonical_df["desc_canonical"].tolist(),
        dim=dim, 
        batch_size=batch_size, 
        task_type=task_type, 
        normalize=normalize,
    )
    
    # Attach the resulting vectors to the dataframe
    canonical_df["embedding"] = list(embeddings_matrix)

    # Save to cache to prevent paying for API calls again
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        canonical_df.to_parquet(cache_path, index=False)
        print(f"Embeddings cached to {cache_path}.")
        
    return canonical_df


def embeddings_as_matrix(emb_df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """
    Helper function for downstream models.
    Splits the embedding DataFrame into two pure data structures:
    1. A list of StockCodes.
    2. A continuous NumPy 2D array (matrix) of the embeddings, ready for clustering or neural networks.
    """
    sku_list = emb_df["StockCode"].astype(str).tolist()
    # Stack the individual vector lists into a single continuous mathematical matrix
    embeddings_matrix = np.vstack(emb_df["embedding"].to_list()).astype(np.float32)
    return sku_list, embeddings_matrix
