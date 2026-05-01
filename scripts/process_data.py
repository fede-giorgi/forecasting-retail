import os
import sys

# Allow running from anywhere by adding the project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np

# Import our specialized pipeline tools via the centralized interface
from src.tools import (
    load_raw_data,
    clean_and_split_transactions,
    aggregate_weekly_sku,
    add_temporal_features,
    add_historical_features,
    add_pricing_features,
    calculate_demand_profile,
    calculate_commercial_profile,
    create_seasonal_profile_clusters,
    create_volume_clusters,
    create_semantic_clusters,
    embed_sku_descriptions
)


def process_data(input_path: str, output_path: str, test_cutoff: str = "2011-09-01"):
    """
    Main orchestration pipeline for the Forecasting Retail project.
    Transforms raw transactional data into a fully featured dataset ready for ML modeling.

    The pipeline executes the following 10 steps:
    1.  **Data Loading**: Reads the raw Excel export into a structured DataFrame.
    2.  **Cleaning & Separation**: Removes noise and splits valid Sales from Returns/Cancellations.
    3.  **Weekly Aggregation**: Groups transactions into weekly buckets with continuous zero-filling.
    4.  **Feature Engineering**: Injects temporal (holidays), historical (lags), and pricing features.
    5.  **Semantic Embeddings**: Uses Gemini API to convert text descriptions into mathematical vectors.
    6.  **Train/Test Split**: Slices the timeline to isolate training data for bias-free profiling.
    7.  **Static SKU Profiles**: Calculates demand intermittency (ADI/CV2) and commercial metrics.
    8.  **Clustering**: Groups products by seasonal shape, semantic meaning, and volumetric tiers.
    9.  **Final Joins**: Maps all computed features and clusters back to the full (Train+Test) panel.
    10. **Export**: Saves the enriched, ML-ready dataset into an optimized Parquet file.

    Args:
        input_path (str): Absolute path to the raw Online Retail II Excel file.
        output_path (str): Destination path for the processed Parquet dataset.
        test_cutoff (str): Date string (YYYY-MM-DD) used to separate data for profiling 
                           to prevent data leakage.

    Returns:
        None: The function exports the resulting DataFrame directly to the specified output_path.
    """
    
    # 1. Load Data
    print("Loading raw retail data (this might take a minute due to Excel format)...")
    raw_df = load_raw_data(input_path)
    
    # 2. Cleaning & Separation
    print("Separating valid sales from returns and cleaning text...")
    sales_df, returns_df = clean_and_split_transactions(raw_df)
    
    # 3. Aggregation to Weekly Frequency
    print("Aggregating transactions into weekly buckets per SKU (with zero-filling)...")
    weekly_sales = aggregate_weekly_sku(sales_df)
    
    # 4. Feature Engineering
    print("Adding temporal (calendar) features...")
    weekly_sales = add_temporal_features(weekly_sales)
    
    print("Adding historical (lags & rolling return rates) features...")
    weekly_sales = add_historical_features(weekly_sales, returns_df)
    
    print("Adding pricing metrics (median price & promotional flags)...")
    weekly_sales = add_pricing_features(weekly_sales, sales_df)
    
    # 5. Semantic Embeddings (For Profiling)
    print("Generating or loading semantic embeddings from product descriptions...")
    embeddings_cache_path = os.path.join(PROJECT_ROOT, "data", "embeddings_cache.parquet")
    embeddings_df = embed_sku_descriptions(sales_df, cache_path=embeddings_cache_path)
    
    # 6. Train/Test Split for Static Profiling (Prevent Data Leakage)
    print("Splitting data to calculate clusters using ONLY training data...")
    weekly_sales_train = weekly_sales[weekly_sales["Week"] < test_cutoff]
    sales_df_train = sales_df[sales_df["InvoiceDate"] < test_cutoff]
    
    # 7. Static SKU Profiles Demand and Commercial (Strictly on Training Data)
    print("Building static SKU profiles (Demand & Commercial) on Train set...")
    demand_df = calculate_demand_profile(weekly_sales_train)
    commercial_df = calculate_commercial_profile(sales_df_train)
    
    # 8. Clustering (Strictly on Training Data)
    print("Creating Behavioral Clusters (52-Week Seasonal Profiles)...")
    profile_clusters = create_seasonal_profile_clusters(weekly_sales_train, n_clusters=4)
    
    print("Creating Semantic Categories from Text Embeddings...")
    semantic_clusters = create_semantic_clusters(embeddings_df, n_clusters=15)
    
    print("Creating Volume Clusters (Jenks Natural Breaks) on Train set...")
    volume_clusters = create_volume_clusters(weekly_sales_train, n_tiers=3)
    
    # 9. Final Joins (Mapping clusters and profiles back to the FULL dataset)
    print("Joining clusters and profiles back to the main weekly panel (Train + Test)...")
    final_df = weekly_sales.merge(profile_clusters, on="StockCode", how="left")
    final_df = final_df.merge(volume_clusters[["StockCode", "volume_cluster_id", "volume_tier"]], on="StockCode", how="left")
    final_df = final_df.merge(semantic_clusters, on="StockCode", how="left")
    final_df = final_df.merge(demand_df, on="StockCode", how="left")
    final_df = final_df.merge(commercial_df, on="StockCode", how="left")
    
    # Drop rows that didn't get clustered (e.g., products that only appeared in the test set or lacked text descriptions)
    final_df = final_df.dropna(subset=["profile_cluster_id", "volume_cluster_id", "semantic_cluster_id"])
    
    # 10. Export
    print(f"Exporting fully featured dataset to {output_path}...")
    final_df.to_parquet(output_path, index=False)
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    # Point to the actual Retail dataset!
    input_p = os.path.join(PROJECT_ROOT, "data", "online_retail_II.xlsx")
    output_p = os.path.join(PROJECT_ROOT, "data", "processed_retail_data.parquet")
    
    # Create the data folder if it doesn't exist
    os.makedirs(os.path.dirname(output_p), exist_ok=True)
    
    process_data(input_p, output_p)