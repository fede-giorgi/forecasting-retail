"""
Advanced Clustering Module for Retail Forecasting.

This module implements a dual-clustering architecture:
1. Profile Clustering: Groups SKUs based on semantic descriptions, demand intermittence, 
   and commercial behavior. This is used to train Pooled (Global-Local) Models.
2. Volume Clustering: Groups SKUs based on absolute sales volume using Natural Breaks 
   (variance minimization), which is highly effective for extremely skewed retail data.
"""

from typing import Iterable
import numpy as np
import pandas as pd
import jenkspy
from sklearn.cluster import KMeans
from .embeddings import embeddings_as_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import hdbscan
from sklearn.preprocessing import StandardScaler


# ---------- Static Metric Extractors ----------

def calculate_demand_profile(weekly_aggregated_sales: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Syntetos-Boylan demand classification metrics for each SKU.
    
    Metrics:
    - ADI (Average Demand Interval): Average weeks between non-zero sales.
    - CV2 (Squared Coefficient of Variation): Volatility of non-zero sales.
    
    Args:
        weekly_aggregated_sales (pd.DataFrame): The weekly aggregated panel.

    Returns:
        pd.DataFrame: A DataFrame containing ADI, CV2, demand_class, and share_zero_weeks.
    """
    profile_rows = []
    
    # Group by StockCode to calculate metrics per product
    for sku, group in weekly_aggregated_sales.groupby("StockCode"):
        sales_series = group.sort_values("Week")["Quantity"].values
        total_weeks = len(sales_series)
        
        # Isolate weeks where the product actually sold
        non_zero_sales = sales_series[sales_series > 0]
        
        # ADI calculation
        adi = total_weeks / max(len(non_zero_sales), 1)
        
        # CV2 calculation
        if len(non_zero_sales) > 1:
            cv2 = (np.std(non_zero_sales) / np.mean(non_zero_sales)) ** 2
        else:
            cv2 = 0.0
            
        # Syntetos-Boylan Classification Matrix
        if adi < 1.32 and cv2 < 0.49:
            demand_class = "smooth"
        elif adi < 1.32:
            demand_class = "erratic"
        elif cv2 < 0.49:
            demand_class = "intermittent"
        else:
            demand_class = "lumpy"
            
        share_zero_weeks = float((sales_series == 0).mean())
        
        profile_rows.append((sku, adi, cv2, demand_class, share_zero_weeks))
        
    return pd.DataFrame(
        profile_rows, 
        columns=["StockCode", "ADI", "CV2", "demand_class", "share_zero_weeks"]
    )


def calculate_commercial_profile(raw_sales: pd.DataFrame) -> pd.DataFrame:
    """
    Computes static commercial features for each SKU.
    
    These features represent the general business footprint of the product, 
    such as how much it costs, how many people buy it at once, and its global reach.
    """
    sku_groups = raw_sales.groupby("StockCode")
    
    commercial_df = pd.DataFrame({
        "price_median": sku_groups["Price"].median(),
        "mean_basket_size": sku_groups["Quantity"].mean(),
        "n_unique_customers": sku_groups["Customer ID"].nunique(),
        "country_uk_share": sku_groups["Country"].apply(lambda series: float((series == "United Kingdom").mean())),
    }).reset_index()
    
    # We drop price_tier from here since we will build a dedicated Volume cluster later
    return commercial_df


# ---------- Dual Clustering Algorithms ----------

def create_profile_clusters(
    embeddings_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    commercial_df: pd.DataFrame,
    weekly_aggregated_sales: pd.DataFrame | None = None,
    umap_dimensions: int = 3,
    n_profile_clusters: int = 4,
    demand_columns: Iterable[str] | None = None,
    commercial_columns: Iterable[str] | None = None,
    random_state: int = 42,
    plot: bool = False,
) -> pd.DataFrame:
    """
    Creates "Behavioral" clusters by fusing Text Embeddings, Demand shape, and Commercial stats.
    
    Uses UMAP to compress text embeddings and KMeans to find distinct behavioral archetypes.
    If weekly_aggregated_sales is provided and plot=True, it also visualizes the timeline.
    """

    demand_columns = list(demand_columns) if demand_columns else ["ADI", "CV2", "share_zero_weeks"]
    commercial_columns = list(commercial_columns) if commercial_columns else [
        "price_median", "mean_basket_size", "n_unique_customers", "country_uk_share"
    ]

    # 1. Inner-join all three dataframes to ensure we only cluster fully represented SKUs
    merged_data = (
        embeddings_df[["StockCode", "embedding"]]
        .merge(demand_df[["StockCode"] + demand_columns], on="StockCode")
        .merge(commercial_df[["StockCode"] + commercial_columns], on="StockCode")
    )

    skus, text_embeddings = embeddings_as_matrix(merged_data)

    # 2. UMAP Dimensionality Reduction on text embeddings
    print("Compressing text embeddings via UMAP...")
    reducer = umap.UMAP(n_components=umap_dimensions, metric="cosine", random_state=random_state)
    embeddings_reduced = reducer.fit_transform(text_embeddings)

    # 3. Standardize the numeric features independently
    print("Scaling and weighting numeric features...")
    embeddings_scaled = StandardScaler().fit_transform(embeddings_reduced)
    demand_scaled = StandardScaler().fit_transform(merged_data[demand_columns].fillna(0).values)
    commercial_scaled = StandardScaler().fit_transform(merged_data[commercial_columns].fillna(0).values)

    # Balance the mathematical weight of the three domains so KMeans considers them equally
    demand_scaled = demand_scaled * 1.5
    commercial_scaled = commercial_scaled * 1.0
    embeddings_scaled = embeddings_scaled * 1.0

    # Concatenate all three worlds into a single feature matrix
    final_feature_matrix = np.concatenate([embeddings_scaled, demand_scaled, commercial_scaled], axis=1)

    # 4. KMeans Clustering
    print(f"Running KMeans to find {n_profile_clusters} distinct profile archetypes...")
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_profile_clusters, random_state=random_state, n_init="auto")
    cluster_labels = kmeans.fit_predict(final_feature_matrix)

    if plot:
        plot_df = merged_data.copy()
        plot_df["profile_cluster_id"] = cluster_labels
        
        # Plot 1: Syntetos-Boylan Space
        if len(plot_df) > 0:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(
                data=plot_df, 
                x='ADI', 
                y='CV2', 
                hue='profile_cluster_id', 
                palette='tab10', 
                alpha=0.6,
                edgecolor=None,
                s=50
            )

            plt.axvline(x=1.32, color='black', linestyle='--', alpha=0.5, label='ADI = 1.32')
            plt.axhline(y=0.49, color='red', linestyle='--', alpha=0.5, label='CV² = 0.49')

            plt.title('Behavioral Clusters mapped on Syntetos-Boylan Space', fontsize=14, fontweight='bold')
            plt.xlabel('Average Demand Interval (ADI)', fontsize=12)
            plt.ylabel('Squared Coefficient of Variation (CV²)', fontsize=12)
            plt.legend(title='Profile Cluster ID', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)

            plt.xlim(0, plot_df['ADI'].quantile(0.98))
            plt.ylim(0, plot_df['CV2'].quantile(0.98))

            plt.tight_layout()
            plt.show()
            
        # Plot 2: Chronological Sales Profiles by Cluster
        if weekly_aggregated_sales is not None:
            # Pivot to get Week on columns, StockCode on index
            pivot_df = weekly_aggregated_sales.pivot_table(index="StockCode", columns="Week", values="Quantity", aggfunc="sum").fillna(0)
            
            # Merge with cluster labels
            pivot_df = pivot_df.merge(pd.DataFrame({"StockCode": skus, "profile_cluster_id": cluster_labels}), on="StockCode").set_index("StockCode")
            
            # Normalize each SKU's timeline by its maximum value so we only compare the "shape" of the curve
            max_vals = pivot_df.drop(columns=["profile_cluster_id"]).max(axis=1) + 1e-9
            normalized_profiles = pivot_df.drop(columns=["profile_cluster_id"]).div(max_vals, axis=0)
            normalized_profiles["profile_cluster_id"] = pivot_df["profile_cluster_id"]
            
            plt.figure(figsize=(12, 6))
            for cluster_id in sorted(normalized_profiles["profile_cluster_id"].unique()):
                cluster_data = normalized_profiles[normalized_profiles["profile_cluster_id"] == cluster_id]
                cluster_mean = cluster_data.drop(columns=["profile_cluster_id"]).mean(axis=0)
                plt.plot(cluster_mean.index, cluster_mean.values, label=f"Cluster {cluster_id} (n={len(cluster_data)})", linewidth=2)
                
            plt.title("Normalized Weekly Sales Profiles by Behavioral Cluster", fontsize=14, fontweight="bold")
            plt.xlabel("Week", fontsize=12)
            plt.ylabel("Normalized Avg Quantity (0 to 1)", fontsize=12)
            plt.legend(title="Profile Cluster ID", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

    return pd.DataFrame({"StockCode": skus, "profile_cluster_id": cluster_labels})


def create_volume_clusters(weekly_aggregated_sales: pd.DataFrame, n_tiers: int = 3, plot: bool = False) -> pd.DataFrame:
    """
    Creates Volume-based clusters to distinguish Top Sellers from Long-Tail products.
    
    Uses Jenks Natural Breaks optimization (via the jenkspy library) to minimize 
    within-class variance. This is highly effective for extremely Pareto-skewed 
    retail sales data, finding the natural "gaps" in sales volumes rather than 
    forcing an equal number of products into each bucket.
    """
    
    # Calculate the total volume sold for each SKU over its entire history
    total_volume_per_sku = weekly_aggregated_sales.groupby("StockCode", as_index=False)["Quantity"].sum()
    
    # Extract the 1D list of volumes, strictly as Python floats and dropping any NaNs
    volumes_list = [float(x) for x in total_volume_per_sku["Quantity"].dropna().tolist()]
    
    try:
        # Check if we have enough unique values for Jenks
        if len(set(volumes_list)) < n_tiers:
            raise ValueError(f"Not enough unique volumes ({len(set(volumes_list))}) for {n_tiers} tiers.")
            
        # Apply Jenks Natural Breaks to find the optimal bin edges
        print(f"Applying Jenks Natural Breaks to segment volumes into {n_tiers} tiers...")
        import jenkspy
        breaks = jenkspy.jenks_breaks(volumes_list, n_classes=n_tiers)
        
        labels = list(range(n_tiers))
        total_volume_per_sku["volume_cluster_id"] = pd.cut(
            total_volume_per_sku["Quantity"], 
            bins=breaks, 
            labels=labels, 
            include_lowest=True
        ).astype(int)
        
    except Exception as e:
        print(f"jenkspy failed ({e}). Falling back to 1D KMeans (mathematical equivalent)...")
        volumes_array = total_volume_per_sku["Quantity"].fillna(0).values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_tiers, random_state=42, n_init="auto")
        cluster_labels = kmeans.fit_predict(volumes_array)
        
        cluster_centers = kmeans.cluster_centers_.flatten()
        sorted_cluster_order = np.argsort(cluster_centers)
        label_mapping = {old_label: new_logical_label for new_logical_label, old_label in enumerate(sorted_cluster_order)}
        total_volume_per_sku["volume_cluster_id"] = [label_mapping[label] for label in cluster_labels]
    
    # Label formatting for readability (e.g. 0 -> 'Low', 1 -> 'Medium', 2 -> 'High' if n_tiers=3)
    if n_tiers == 3:
        tier_names = {0: "Low", 1: "Medium", 2: "High"}
        total_volume_per_sku["volume_tier"] = total_volume_per_sku["volume_cluster_id"].map(tier_names)
        
    if plot and n_tiers == 3:
        plot_vol_df = total_volume_per_sku[total_volume_per_sku['Quantity'] > 0]
        pcts = plot_vol_df['volume_tier'].value_counts(normalize=True) * 100
        colors = {'Low': '#4CAF93', 'Medium': '#F0A500', 'High': '#E05C5C'}

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle('SKU Volumetric Segments — Jenks Breakpoints (Log Scale)', fontsize=14, fontweight='bold')

        for label, color in colors.items():
            if label not in plot_vol_df['volume_tier'].values:
                continue
                
            vals = np.log1p(plot_vol_df.loc[plot_vol_df['volume_tier'] == label, 'Quantity'])
            ax.hist(
                vals, 
                bins=50, 
                color=color, 
                alpha=0.65, 
                label=f"{label} ({pcts.get(label, 0):.1f}%)", 
                edgecolor='white', 
                linewidth=0.4
            )

        low_max = plot_vol_df[plot_vol_df['volume_tier'] == 'Low']['Quantity'].max()
        med_max = plot_vol_df[plot_vol_df['volume_tier'] == 'Medium']['Quantity'].max()

        ax.axvline(np.log1p(low_max), color='black', linestyle='--', linewidth=1.2, label=f'Low/Medium edge: {low_max:,.0f} units')
        ax.axvline(np.log1p(med_max), color='black', linestyle=':', linewidth=1.2, label=f'Medium/High edge: {med_max:,.0f} units')

        tick_vals_qty = [1, 10, 100, 1000, 10000, 50000, 100000]
        ax.set_xticks(np.log1p(tick_vals_qty))
        ax.set_xticklabels([f"{v:,}" for v in tick_vals_qty])

        ax.set_xlabel('Historical Total Quantity Sold (Log Scale)', fontsize=12)
        ax.set_ylabel('Density Count (Number of SKUs)', fontsize=12)
        ax.legend(fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)

        plt.tight_layout()
        plt.show()
        
    # Return just the mapping to be used as a feature
    return_columns = ["StockCode", "volume_cluster_id", "volume_tier"] if n_tiers == 3 else ["StockCode", "volume_cluster_id"]
    return total_volume_per_sku[return_columns]
