import os
import sys

# Allow running from anywhere by adding the project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np

# Import all our beautifully refactored modules
from src.tools.data_loader import load_raw_data
from src.tools.cleaning import split_sales_returns
from src.tools.add_temporal_features import add_temporal_features
from src.tools.feature_engineering import aggregate_weekly_sku, add_historical_features, add_pricing_features
from src.tools.clustering import calculate_demand_profile, calculate_commercial_profile, create_seasonal_profile_clusters, create_volume_clusters, create_semantic_clusters
from src.tools.embeddings import embed_sku_descriptions


def process_data(input_path: str, output_path: str, test_cutoff: str = "2011-09-01"):
    """
    Main orchestration pipeline for the Forecasting Retail project.
    Transforms raw transactional data into a fully featured dataset ready for ML modeling.
    """
    
    # 1. Load Data
    print("Loading raw retail data (this might take a minute due to Excel format)...")
    raw_df = load_raw_data(input_path)
    
    # 2. Cleaning & Separation
    print("Separating valid sales from returns and cleaning text...")
    sales_df, returns_df = split_sales_returns(raw_df)
    
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
    embeddings_cache_path = os.path.join(PROJECT_ROOT, "Datasets", "embeddings_cache.parquet")
    embeddings_df = embed_sku_descriptions(sales_df, cache_path=embeddings_cache_path)
    
    # 6. Train/Test Split for Static Profiling (Prevent Data Leakage)
    print("Splitting data to calculate clusters using ONLY training data...")
    weekly_sales_train = weekly_sales[weekly_sales["Week"] < test_cutoff]
    sales_df_train = sales_df[sales_df["InvoiceDate"] < test_cutoff]
    
    # 7. Clustering (Strictly on Training Data)
    print("Building static SKU profiles (Demand & Commercial) on Train set...")
    demand_df = calculate_demand_profile(weekly_sales_train)
    commercial_df = calculate_commercial_profile(sales_df_train)
    
    print("Creating Behavioral Clusters (52-Week Seasonal Profiles)...")
    profile_clusters = create_seasonal_profile_clusters(weekly_sales_train, n_clusters=4)
    
    print("Creating Semantic Categories from Text Embeddings...")
    semantic_clusters = create_semantic_clusters(embeddings_df, n_clusters=15)
    
    print("Creating Volume Clusters (Jenks Natural Breaks) on Train set...")
    volume_clusters = create_volume_clusters(weekly_sales_train, n_tiers=3)
    
    # 8. Final Joins (Mapping clusters back to the FULL dataset)
    print("Joining clusters back to the main weekly panel (Train + Test)...")
    final_df = weekly_sales.merge(profile_clusters, on="StockCode", how="left")
    final_df = final_df.merge(volume_clusters[["StockCode", "volume_cluster_id", "volume_tier"]], on="StockCode", how="left")
    final_df = final_df.merge(semantic_clusters, on="StockCode", how="left")
    
    # Drop rows that didn't get clustered (e.g., products that only appeared in the test set or lacked text descriptions)
    final_df = final_df.dropna(subset=["profile_cluster_id", "volume_cluster_id", "semantic_cluster_id"])
    
    # 8. Export
    print(f"Exporting fully featured dataset to {output_path}...")
    final_df.to_parquet(output_path, index=False)
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    # Point to the actual Retail dataset!
    input_p = os.path.join(PROJECT_ROOT, "Datasets", "online_retail_II.xlsx")
    output_p = os.path.join(PROJECT_ROOT, "Datasets", "processed_retail_data.parquet")
    
    # Create the Datasets folder if it doesn't exist
    os.makedirs(os.path.dirname(output_p), exist_ok=True)
    
    process_data(input_p, output_p)