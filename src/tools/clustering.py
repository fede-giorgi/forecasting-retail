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

def create_seasonal_profile_clusters(
    weekly_aggregated_sales: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42,
    plot: bool = False,
) -> pd.DataFrame:
    """
    Identifies consumption shapes (profiles) based ONLY on the seasonal timeline.
    Extracts the 52-week average sales profile for each SKU, scales it (MinMax) 
    to remove volume bias, and clusters the pure "shape" using KMeans.
    """
    print(f"Calculating seasonal shape clusters (k={n_clusters}) based on weekly data...")
    
    df = weekly_aggregated_sales.copy()
    
    # Extract week of the year (1 to 52/53)
    df["week_of_year"] = df["Week"].dt.isocalendar().week
    
    # Calculate the average quantity per week-of-year across all years
    profiles = df.groupby(["StockCode", "week_of_year"], observed=True)["Quantity"].mean().unstack().fillna(0)
    
    # Ensure all weeks from 1 to 52 are represented (just in case)
    profiles = profiles.reindex(columns=range(1, 53), fill_value=0)
    
    # Scale each SKU (row) individually to be between 0 and 1
    # This isolates the "shape" of the seasonality, removing the effect of absolute volume
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    profiles_scaled = scaler.fit_transform(profiles.T).T
    
    # Fit KMeans
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(profiles_scaled)
    
    if plot:
        print(f"Executing final K-Means with k={n_clusters} clusters...")
        
        plt.figure(figsize=(12, 6))
        for i in range(n_clusters):
            # Calculate the mean profile for the entire cluster
            cluster_mean = profiles_scaled[cluster_labels == i].mean(axis=0)
            cluster_size = sum(cluster_labels == i)
            plt.plot(range(1, 53), cluster_mean, label=f'Cluster {i} (n={cluster_size})', linewidth=2)

        plt.title('Normalized 52-Week Seasonal Profiles by SKU Cluster', fontsize=14, fontweight='bold')
        plt.xlabel('Week of the Year', fontsize=12)
        plt.ylabel('Normalized Average Quantity (0 to 1)', fontsize=12)
        plt.xticks(range(1, 53, 4)) # Tick every 4 weeks (roughly a month)
        plt.legend(title="Seasonal Profile Cluster")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return pd.DataFrame({"StockCode": profiles.index, "profile_cluster_id": cluster_labels})


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


def create_semantic_clusters(
    embeddings_df: pd.DataFrame,
    n_clusters: int = 15,
    n_keywords: int = 3,
    random_state: int = 42,
    plot: bool = False
) -> pd.DataFrame:
    """
    Clusters products purely based on their Gemini semantic text embeddings.
    Extracts human-readable names for each cluster using TF-IDF on the descriptions.
    """
    print(f"Clustering semantic embeddings into {n_clusters} distinct categories...")
    skus, text_embeddings = embeddings_as_matrix(embeddings_df)
    
    # 1. Cluster the raw 768-D embeddings (they are already L2 normalized, so Euclidean/KMeans works well)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    cluster_labels = kmeans.fit_predict(text_embeddings)
    
    df = embeddings_df.copy()
    df["semantic_cluster_id"] = cluster_labels
    
    # 2. Extract Keywords for each cluster using TF-IDF
    print("Extracting cluster names via TF-IDF...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    cluster_names = {}
    
    for cluster_id in range(n_clusters):
        docs = df[df["semantic_cluster_id"] == cluster_id]["desc_canonical"].dropna().tolist()
        if not docs:
            cluster_names[cluster_id] = f"Cluster {cluster_id}"
            continue
            
        # Fit TF-IDF on the descriptions of this cluster
        vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        try:
            tfidf_matrix = vectorizer.fit_transform(docs)
            # Sum tf-idf scores across all docs in the cluster
            word_scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
            words = vectorizer.get_feature_names_out()
            
            # Get top N keywords
            top_indices = word_scores.argsort()[-n_keywords:][::-1]
            top_words = [words[i].upper() for i in top_indices]
            cluster_names[cluster_id] = " + ".join(top_words)
        except ValueError:
            # Fallback if TF-IDF fails (e.g. only numbers/stopwords)
            cluster_names[cluster_id] = f"Cluster {cluster_id}"
            
    df["semantic_cluster_name"] = df["semantic_cluster_id"].map(cluster_names)
    
    # 3. Plotting
    if plot:
        print("Reducing dimensions via UMAP for semantic visualization...")
        import umap
        reducer = umap.UMAP(n_components=2, metric="cosine", random_state=random_state)
        umap_2d = reducer.fit_transform(text_embeddings)
        
        plot_df = pd.DataFrame({
            "UMAP1": umap_2d[:, 0],
            "UMAP2": umap_2d[:, 1],
            "Cluster Name": df["semantic_cluster_name"]
        })
        
        plt.figure(figsize=(14, 8))
        sns.scatterplot(
            data=plot_df, 
            x="UMAP1", 
            y="UMAP2", 
            hue="Cluster Name", 
            palette="tab20", 
            alpha=0.7, 
            edgecolor=None,
            s=40
        )
        plt.title('Semantic Product Categories (UMAP 2D Projection)', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Semantic Clusters", fontsize=9)
        plt.tight_layout()
        plt.show()
        
    return df[["StockCode", "semantic_cluster_id", "semantic_cluster_name"]]

