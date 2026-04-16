import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# Human-readable segment names mapped from cluster centroids
SEGMENT_NAMES = {
    0: "Low Value Customers",
    1: "Potential Customers",
    2: "Loyal Customers",
    3: "High Value Customers",
}


def run_kmeans(rfm_df, n_clusters=4):
    """
    Run K-Means clustering on scaled RFM features.
    Returns rfm_df with a new 'Segment' column.
    """
    features = rfm_df[["Recency", "Frequency", "Monetary"]].copy()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm_df = rfm_df.copy()
    rfm_df["Cluster"] = kmeans.fit_predict(scaled)

    # Rank clusters by Monetary mean to assign consistent labels
    cluster_monetary = (
        rfm_df.groupby("Cluster")["Monetary"].mean().sort_values()
    )
    rank_map = {
        cluster: idx for idx, cluster in enumerate(cluster_monetary.index)
    }
    rfm_df["Segment"] = rfm_df["Cluster"].map(rank_map).map(SEGMENT_NAMES)

    return rfm_df


def segment_summary(rfm_df):
    """Return a summary dict of segment counts and percentages."""
    counts = rfm_df["Segment"].value_counts()
    total = counts.sum()
    summary = [
        {
            "segment": seg,
            "count": int(cnt),
            "pct": round(cnt / total * 100, 1),
        }
        for seg, cnt in counts.items()
    ]
    return summary
