# Support Vector Machines (SVM) & K-Nearest Neighbors (KNN)

## Concept
Support Vector Machines (SVM) and K-Nearest Neighbors (KNN) represent two fundamentally different geometric paradigms
in machine learning: max-margin optimization (a global, convex objective) vs local instance-based memory learning.

1. **Support Vector Machines (SVM)**: Formulate classification as finding the unique hyperplane that maximizes
   the geometric margin between positive and negative classes. When data is non-linearly separable, SVM leverages
   the **Kernel Trick** to implicitly project data into a higher-dimensional (possibly infinite-dimensional) feature space where a linear boundary may exist.
2. **K-Nearest Neighbors (KNN)**: A non-parametric, lazy-learning algorithm that memorizes training samples.
   At inference time, query instances are classified by computing distances to all stored points and taking the majority vote.

---

## Formula / Core Idea

### 1. Hard-Margin & Soft-Margin SVM Formulations
Given linearly separable training data $(\mathbf{x}_i, y_i)$ where $y_i \in \{-1, +1\}$, the geometric margin is $\frac{2}{\|\mathbf{w}\|}$.
Maximizing margin is equivalent to minimizing $\frac{1}{2} \|\mathbf{w}\|^2$ subject to $y_i(\mathbf{w}^T \mathbf{x}_i + b) \ge 1$.

For real-world non-separable data, we introduce slack variables $\xi_i \ge 0$ and regularization parameter $C > 0$:
$$\min_{\mathbf{w}, b, \xi} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^N \xi_i \quad \text{s.t.} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i$$

- Large $C$: Penalizes margin violations strictly (narrow margin, risks overfitting).
- Small $C$: Tolerates margin violations (wide margin, higher bias, more robust to noise).

### 2. The Kernel Trick & Mercer's Theorem
The dual formulation of SVM depends only on pairwise inner products $\langle \mathbf{x}_i, \mathbf{x}_j \rangle$.
The Kernel Trick replaces inner products with a kernel function $K(\mathbf{x}_i, \mathbf{x}_j) = \langle \phi(\mathbf{x}_i), \phi(\mathbf{x}_j) \rangle$:
- **Linear Kernel**: $K(\mathbf{u}, \mathbf{v}) = \mathbf{u}^T \mathbf{v}$
- **Polynomial Kernel**: $K(\mathbf{u}, \mathbf{v}) = (\gamma \mathbf{u}^T \mathbf{v} + c)^d$
- **Radial Basis Function (RBF / Gaussian)**: $K(\mathbf{u}, \mathbf{v}) = \exp(-\gamma \|\mathbf{u} - \mathbf{v}\|^2)$

### 3. K-Nearest Neighbors (KNN) Distance Metrics
The Minkowski distance of order $p$ between vectors $\mathbf{u}$ and $\mathbf{v}$:
$$D_p(\mathbf{u}, \mathbf{v}) = \left( \sum_{i=1}^D |u_i - v_i|^p \right)^{1/p}$$
- $p = 1$: Manhattan ($L_1$) distance.
- $p = 2$: Euclidean ($L_2$) distance.

---

## When to Use

- **SVM**: Effective for medium-sized, high-dimensional datasets (e.g. bioinformatics, text classification, handwriting recognition).
- **KNN**: Simple baseline for low-dimensional recommendation systems, nearest-neighbor search, and imputing missing values (`KNNImputer`).

---

## Example

Below is a Python demonstration comparing Non-linear RBF SVM against KNN on an intertwined non-linear dataset:

```python
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 1. Synthesize non-linearly separable dataset
X, y = make_moons(n_samples=500, noise=0.2, random_state=42)

# 2. Build standardized pipelines
svm_pipeline = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", C=1.0, gamma="scale")
)

knn_pipeline = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5, weights="distance")
)

# 3. Evaluate via 5-Fold Stratified Cross-Validation
svm_scores = cross_val_score(svm_pipeline, X, y, cv=5, scoring="accuracy")
knn_scores = cross_val_score(knn_pipeline, X, y, cv=5, scoring="accuracy")

print(f"RBF Kernel SVM 5-Fold Accuracy: {svm_scores.mean():.4f} (+/- {svm_scores.std():.4f})")
print(f"KNN (k=5) 5-Fold Accuracy:       {knn_scores.mean():.4f} (+/- {knn_scores.std():.4f})")
```

---

## Key Points & Pitfalls

### 1. Scaling is Mandatory for Both SVM and KNN
Both algorithms depend strictly on Euclidean distance calculations. An unscaled feature with a wide range
completely dominates the margin or nearest-neighbor selection.

### 2. SVM Computational Complexity ($O(N^2)$ to $O(N^3)$)
Training kernel SVM scales roughly between quadratic and cubic in the sample size $N$, so it becomes impractically slow beyond roughly $10^5$ samples. For large datasets, use `LinearSVC` or `SGDClassifier`.

### 3. KNN Inference Latency ($O(N \cdot D)$)
KNN has essentially no training cost (it only stores the data, or builds an index), but computing distances against all $N$ points at inference time causes massive latency.
Use KD-Trees, Ball Trees, or Approximate Nearest Neighbor libraries (Faiss, Annoy, hnswlib) for large datasets.

---

