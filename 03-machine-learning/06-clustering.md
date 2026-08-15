# Clustering

## Concept
Unsupervised grouping of similar data points.

## Formula / Core Idea
K-Means: assign points to nearest centroid, update centroids, repeat. Also DBSCAN, hierarchical.

## When to Use
Customer segmentation, anomaly detection, exploring unlabelled data.

## Example
```python
from sklearn.cluster import KMeans
labels = KMeans(n_clusters=3, n_init=10).fit_predict(X)
```

## Key Points
- Choose k with the elbow method or silhouette score
- K-Means assumes round clusters
- DBSCAN finds arbitrary shapes and noise
- Scale features first

## My Notes
- (add your own notes here)
