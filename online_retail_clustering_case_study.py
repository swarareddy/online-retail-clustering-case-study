"""Online Retail Transaction Analysis and Clustering.

Companion script to online_retail_clustering_case_study.ipynb.
Keep "Online retail_new.csv" in the same directory before running.
"""
from IPython.display import display

# Online Retail Transaction Analysis and Clustering
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors

RANDOM_STATE = 42
CSV_PATH = "Online retail_new.csv"  # Keep the CSV in the same folder as this notebook.


# 4. Load and clean the transaction data
raw = pd.read_csv(CSV_PATH, header=None, dtype=str, encoding_errors="replace")
print("Raw shape:", raw.shape)

# Normalize names and treat blank strings as missing values.
raw = raw.apply(lambda col: col.str.strip().str.lower())
raw = raw.replace("", np.nan)

# Keep transactions as lists of distinct products.
transactions = []
for _, row in raw.iterrows():
    items = [str(x).strip() for x in row.dropna().tolist() if str(x).strip()]
    # Duplicate products in a basket do not increase the binary basket count.
    transactions.append(sorted(set(items)))

transactions = [basket for basket in transactions if basket]
all_items = sorted({item for basket in transactions for item in basket})

print("Number of usable transactions:", len(transactions))
print("Number of unique products:", len(all_items))
print("First cleaned transaction:", transactions[0])
print("Empty transactions removed:", len(raw) - len(transactions))

# **Purpose and observation:** This step standardizes capitalization and whitespace so that the same product is not counted under multiple spellings. Empty cells are expected in this dataset because baskets contain different numbers of products. The number of transactions and unique products printed above describe the cleaned dataset.
# 5. Descriptive statistics and product frequency
transaction_sizes = pd.Series([len(basket) for basket in transactions], name="transaction_size")
size_summary = transaction_sizes.describe().to_frame().round(2)
print("Transaction-size descriptive statistics:")
display(size_summary)

product_counts = pd.Series(
    {item: sum(item in basket for basket in transactions) for item in all_items},
    name="transaction_count"
).sort_values(ascending=False)

print("\nTop 15 products by number of baskets containing them:")
display(product_counts.head(15).to_frame())

print("Median basket size:", transaction_sizes.median())
print("Most common basket size:", int(transaction_sizes.mode().iloc[0]))

# Graph 1: most frequent products
top_n = 15
top_products = product_counts.head(top_n).sort_values()
plt.figure(figsize=(10, 6))
top_products.plot(kind="barh")
plt.title(f"Top {top_n} Products by Transaction Frequency")
plt.xlabel("Number of transactions containing product")
plt.ylabel("Product")
plt.tight_layout()
plt.show()

# Graph 2: distribution of transaction sizes
plt.figure(figsize=(9, 5))
bins = range(1, int(transaction_sizes.max()) + 2)
plt.hist(transaction_sizes, bins=bins, edgecolor="black", align="left")
plt.title("Distribution of Transaction Sizes")
plt.xlabel("Number of distinct products in a transaction")
plt.ylabel("Number of transactions")
plt.xticks(range(1, int(transaction_sizes.max()) + 1))
plt.tight_layout()
plt.show()

# **Interpretation guide:** In the first graph, taller bars indicate products appearing in more baskets and may help prioritize shelf availability or promotions. The histogram shows whether baskets are usually small or large. Use the printed statistics and graph shape to write the dataset-specific observation; do not infer that a frequently purchased item causes another item to be purchased.
# 6. Convert baskets into numerical feature vectors
item_to_col = {item: i for i, item in enumerate(all_items)}
X = np.zeros((len(transactions), len(all_items)), dtype=np.uint8)
for r, basket in enumerate(transactions):
    for item in basket:
        X[r, item_to_col[item]] = 1

basket_matrix = pd.DataFrame(X, columns=all_items)
print("Transaction-item matrix shape:", basket_matrix.shape)
print("Matrix density (% of entries equal to 1):", round(100 * X.mean(), 3))
display(basket_matrix.iloc[:5, :10])

# 7. PCA for dimensionality reduction and visualization
# PCA on binary item vectors; this dataset is small enough for dense PCA.
pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca_2d = pca_2d.fit_transform(X)
explained = pca_2d.explained_variance_ratio_

print("Variance explained by PC1:", round(explained[0] * 100, 2), "%")
print("Variance explained by PC2:", round(explained[1] * 100, 2), "%")
print("Combined variance explained:", round(explained.sum() * 100, 2), "%")

plt.figure(figsize=(9, 6))
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], s=10, alpha=0.35)
plt.title("PCA Projection of Shopping Transactions")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.show()

# **Interpretation:** Points close together have similar projections in the two-dimensional PCA view. Overlap is expected in sparse transaction data. Report the explained-variance percentages printed above to explain how much of the original variation is represented by this two-dimensional chart.
# 8. Prepare a consistent sample for clustering
SAMPLE_SIZE = min(1200, len(X))
rng = np.random.default_rng(RANDOM_STATE)
sample_idx = np.sort(rng.choice(len(X), size=SAMPLE_SIZE, replace=False))

pca_10 = PCA(n_components=min(10, X.shape[1]), random_state=RANDOM_STATE)
X_pca_all = pca_10.fit_transform(X)
X_sample_pca = X_pca_all[sample_idx]
scaler = StandardScaler()
X_cluster = scaler.fit_transform(X_sample_pca)

print("Transactions used for clustering:", len(X_cluster))
print("Clustering feature dimensions:", X_cluster.shape[1])
print("Cumulative variance explained by these PCA components:",
      round(pca_10.explained_variance_ratio_.sum() * 100, 2), "%")

# 9. Fit four clustering algorithms
N_CLUSTERS = 4
cluster_labels = {}

cluster_labels["K-means"] = KMeans(
    n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10
).fit_predict(X_cluster)

cluster_labels["Hierarchical"] = AgglomerativeClustering(
    n_clusters=N_CLUSTERS, linkage="ward"
).fit_predict(X_cluster)

# Density-based clustering. Noise points are labelled -1.
cluster_labels["DBSCAN"] = DBSCAN(eps=2.5, min_samples=8).fit_predict(X_cluster)

cluster_labels["Spectral"] = SpectralClustering(
    n_clusters=N_CLUSTERS, affinity="nearest_neighbors",
    n_neighbors=10, assign_labels="kmeans", random_state=RANDOM_STATE
).fit_predict(X_cluster)

for name, labels in cluster_labels.items():
    values, counts = np.unique(labels, return_counts=True)
    print(f"{name}: {dict(zip(values.tolist(), counts.tolist()))}")

# 10. Evaluate and compare clustering performance
def evaluate_clustering(X_eval, labels, algorithm):
    labels = np.asarray(labels)
    is_dbscan = algorithm == "DBSCAN"
    if is_dbscan:
        keep = labels != -1
        X_eval = X_eval[keep]
        labels = labels[keep]
        noise_fraction = 1 - keep.mean()
    else:
        noise_fraction = 0.0

    n_clusters = len(np.unique(labels))
    n_eval = len(labels)
    if n_clusters < 2 or n_clusters >= n_eval:
        sil = np.nan
        dbi = np.nan
        note = "Metrics undefined: need 2 or more clusters and fewer clusters than observations."
    else:
        sil = silhouette_score(X_eval, labels)
        dbi = davies_bouldin_score(X_eval, labels)
        note = "Noise excluded from scores." if is_dbscan else "All sampled observations scored."

    return {
        "Algorithm": algorithm,
        "Clusters (excluding noise)": n_clusters,
        "Noise fraction": round(noise_fraction, 3),
        "Silhouette coefficient": sil,
        "Davies-Bouldin index": dbi,
        "Evaluation note": note
    }

results = pd.DataFrame([
    evaluate_clustering(X_cluster, labels, name)
    for name, labels in cluster_labels.items()
])
display(results.round(4))
results.to_csv("clustering_metrics.csv", index=False)

# Graph 3: show cluster assignments in the first two PCA dimensions.
# Labels are from the same sampled transactions used in the comparison.
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, (name, labels) in zip(axes.ravel(), cluster_labels.items()):
    scatter = ax.scatter(
        X_pca_all[sample_idx, 0], X_pca_all[sample_idx, 1],
        c=labels, s=12, alpha=0.65, cmap="tab10"
    )
    ax.set_title(f"{name} clusters in PCA view")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
plt.tight_layout()
plt.show()

# Metric comparison plots (NaN values are naturally omitted).
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
results.set_index("Algorithm")["Silhouette coefficient"].plot(kind="bar", ax=axes[0])
axes[0].set_title("Silhouette Coefficient by Algorithm")
axes[0].set_xlabel("Clustering algorithm")
axes[0].set_ylabel("Silhouette coefficient (higher is better)")
axes[0].tick_params(axis="x", rotation=25)

results.set_index("Algorithm")["Davies-Bouldin index"].plot(kind="bar", ax=axes[1])
axes[1].set_title("Davies–Bouldin Index by Algorithm")
axes[1].set_xlabel("Clustering algorithm")
axes[1].set_ylabel("Davies–Bouldin index (lower is better)")
axes[1].tick_params(axis="x", rotation=25)
plt.tight_layout()
plt.show()

# 11. Interpreting the results
# Summarize most common products within each non-noise cluster.
sample_baskets = [transactions[i] for i in sample_idx]
cluster_product_rows = []
for name, labels in cluster_labels.items():
    for label in sorted(set(labels)):
        if label == -1:
            continue
        basket_ids = np.where(labels == label)[0]
        counts = {}
        for idx in basket_ids:
            for item in sample_baskets[idx]:
                counts[item] = counts.get(item, 0) + 1
        top = sorted(counts.items(), key=lambda pair: pair[1], reverse=True)[:5]
        cluster_product_rows.append({
            "Algorithm": name,
            "Cluster": int(label),
            "Transactions": len(basket_ids),
            "Top products": ", ".join([f"{item} ({count})" for item, count in top])
        })

cluster_summary = pd.DataFrame(cluster_product_rows)
display(cluster_summary)
cluster_summary.to_csv("cluster_product_summary.csv", index=False)

# 12. Limitations