# Unsupervised Clustering: K-Means, DBSCAN & Hierarchical

## Concept
Clustering is the primary unsupervised learning paradigm for discovering latent natural groupings
in unlabeled data. Unlike supervised classification where ground-truth target classes guide optimization,
clustering algorithms group instances solely based on intrinsic geometric proximity and probability densities.

Clustering algorithms fall into three primary algorithmic classes:
1. **Centroid-Based (K-Means)**: Partitions space into Voronoi cells around $K$ cluster prototypes.
2. **Density-Based (DBSCAN)**: Identifies contiguous dense regions separated by sparse noise areas.
3. **Hierarchical (Agglomerative)**: Builds a nested tree of clusters via iterative pairwise merges.

---

## Formula / Core Idea

### 1. K-Means Objective (Inertia)
K-Means minimizes the Within-Cluster Sum of Squares (WCSS / Inertia):
$$J = \sum_{k=1}^K \sum_{\mathbf{x}_i \in S_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$$

Where $\boldsymbol{\mu}_k = \frac{1}{|S_k|} \sum_{\mathbf{x} \in S_k} \mathbf{x}$ is the centroid of cluster $S_k$.
The standard Lloyd's algorithm alternates between two steps until convergence:
- **Assignment**: $S_k = \{ \mathbf{x}_i : \|\mathbf{x}_i - \boldsymbol{\mu}_k\| \le \|\mathbf{x}_i - \boldsymbol{\mu}_j\|, \forall j \}$
- **Update**: $\boldsymbol{\mu}_k = \frac{1}{|S_k|} \sum_{\mathbf{x} \in S_k} \mathbf{x}$

### 2. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
DBSCAN parameterizes density using two thresholds: radius $\epsilon$ and minimum points `min_samples`.
- **Core Point**: A point with at least `min_samples` within distance $\epsilon$.
- **Border Point**: Not a core point, but within $\epsilon$ of a core point.
- **Noise Outlier**: Neither a core point nor a border point.
Crucially, DBSCAN **does not require specifying $K$ in advance** and can discover arbitrarily shaped, non-convex clusters.

### 3. Clustering Validation: The Silhouette Coefficient
For sample $i$ belonging to cluster $A$:
- $a(i)$: Mean distance from $i$ to all other points in cluster $A$ (cohesion).
- $b(i)$: Mean distance from $i$ to all points in the nearest neighboring cluster $B$ (separation).
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))} \in [-1, +1]$$

- $s(i) \approx +1$: Sample is well-matched to its cluster and far from neighbors.
- $s(i) \approx 0$: Sample is on the decision boundary between two clusters.
- $s(i) < 0$: Sample is assigned to the wrong cluster.

---

## When to Use

- **K-Means**: Best when clusters are roughly spherical, convex, and of similar size and density.
- **DBSCAN**: Best when clusters have arbitrary geometric shapes (crescents, rings), varying densities, or contain significant background noise.
- **Hierarchical**: Best when domain hierarchy is expected (e.g. taxonomy of animal species or organizational structures).

---

## Example

Below is a Python demonstration showing how K-Means fails on non-convex geometric data while DBSCAN succeeds:

```python
from sklearn.datasets import make_circles
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np

# 1. Generate concentric circular clusters
X, _ = make_circles(n_samples=600, factor=0.4, noise=0.05, random_state=42)

# 2. Apply K-Means (Fails because clusters are not linearly separable Voronoi cells)
kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
labels_kmeans = kmeans.fit_predict(X)

# 3. Apply DBSCAN (Succeeds via density reachability)
dbscan = DBSCAN(eps=0.15, min_samples=5)
labels_dbscan = dbscan.fit_predict(X)

n_clusters_db = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
n_noise_db = list(labels_dbscan).count(-1)

print("Clustering Results on Concentric Rings Data:")
print(f"K-Means Inertia: {kmeans.inertia_:.2f}")
print(f"DBSCAN Found Clusters: {n_clusters_db} | Noise Outlier Points: {n_noise_db}")
```

---

## Key Points & Pitfalls

### 1. K-Means Sensitivity to Initialization
Random centroid initialization frequently gets trapped in poor local minima. Always use **K-Means++** (`init='k-means++'`),
which seeds initial centroids probabilistically proportional to squared distance from existing centers.

### 2. Curse of Dimensionality in Clustering
In high-dimensional spaces ($D > 50$), distance metrics concentrate and density vanishes. Run PCA or UMAP to reduce dimensions
before running K-Means or DBSCAN.

### 3. Choosing the Number of Clusters ($K$)
Use the **Elbow Method** (plotting Inertia vs $K$ to find the point of diminishing returns) together with Silhouette Analysis.

---
