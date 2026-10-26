# NumPy: High-Performance Array Computing

## Concept
NumPy (Numerical Python) is the core computational engine underlying virtually all scientific and machine
learning libraries in Python. It provides the homogeneous multidimensional array object (`ndarray`)
together with an extensive collection of compiled routines for fast mathematical, logical, and linear algebra operations.

At its core, NumPy solves Python's scalar processing bottleneck by operating directly on contiguous memory blocks
in C and Fortran. Understanding how `ndarray` strides, memory layouts, and broadcasting rules work
allows machine learning practitioners to write ultra-fast, vectorized algorithms that approach the performance
of native C code without losing Python's syntactic elegance.

---

## Formula / Core Idea

### 1. Anatomy of an `ndarray`
A NumPy array is an abstraction over a raw block of memory (the data buffer). It consists of:
1. **Data Pointer**: Memory address of the first byte.
2. **Data Type (`dtype`)**: Describes the interpretation of bytes (e.g., `float32`, `int64`).
3. **Shape**: A tuple indicating array dimensions $(d_0, d_1, \dots, d_{k-1})$.
4. **Strides**: A tuple indicating the number of bytes to step in memory to advance by 1 element along each dimension.

```text
Logical 2D Array: Shape (3, 4), dtype = int32 (4 bytes per element)
Row 0: [ 0,  1,  2,  3]
Row 1: [ 4,  5,  6,  7]
Row 2: [ 8,  9, 10, 11]

Memory Buffer (C-Order / Row-Major):
[0][1][2][3][4][5][6][7][8][9][10][11] (Contiguous bytes in RAM)
Strides = (16, 4) -> Step 16 bytes to move down a row; Step 4 bytes to move right a column.
```

Because slicing an array merely creates a new array object with modified strides and shapes without
copying the data buffer, basic slices execute in $O(1)$ constant time regardless of array size.

### 2. Broadcasting Semantics
Broadcasting describes how NumPy treats arrays with different shapes during arithmetic operations.
Shapes are compared starting from the trailing (rightmost) dimension. Two dimensions are **compatible** when:
1. They are equal, OR
2. One of them is 1.

If dimensions are unequal and neither is 1, a `ValueError: operands could not be broadcast together` is thrown.
During broadcasting, dimensions of size 1 are virtually expanded (by setting that axis stride to 0) to match the larger shape,
achieving zero-copy memory efficiency.

$$\begin{pmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{pmatrix}_{(2, 2)} + \begin{pmatrix} b_1 & b_2 \end{pmatrix}_{(1, 2)} = \begin{pmatrix} a_{11}+b_1 & a_{12}+b_2 \\ a_{21}+b_1 & a_{22}+b_2 \end{pmatrix}_{(2, 2)}$$

### 3. Essential Linear Algebra Operations
- Matrix Multiplication: $C = A \cdot B \implies$ `C = A @ B` or `np.matmul(A, B)`
- Vector Dot Product: $\mathbf{u}^T \mathbf{v} = \sum_{i=1}^D u_i v_i \implies$ `np.dot(u, v)`
- Euclidean $L_2$ Norm: $\|\mathbf{x}\|_2 = \sqrt{\sum x_i^2} \implies$ `np.linalg.norm(x)`
- Matrix Inverse: $A^{-1} \implies$ `np.linalg.inv(A)`
- Eigen Decomposition: $A \mathbf{v} = \lambda \mathbf{v} \implies$ `eigenvalues, eigenvectors = np.linalg.eig(A)`

---

## When to Use

- **Feature Matrix Preprocessing**: Scaling, standardizing, and transforming numeric feature columns.
- **Vectorized Loss Computations**: Calculating Mean Squared Error (MSE), Cross-Entropy, or Log-Loss.
- **Batch Distance Matrix Calculations**: Computing pairwise Euclidean or cosine distances for clustering and KNN.
- **Synthetic Data Generation**: Sampling multidimensional random tensors from Gaussian, uniform, or binomial distributions.

---

## Example

Here is a complete, vectorized implementation of the **K-Means Clustering** algorithm in pure NumPy:

```python
import numpy as np

class PureNumPyKMeans:
    """Vectorized K-Means clustering algorithm built using NumPy."""
    
    def __init__(self, n_clusters: int = 3, max_iters: int = 100, tol: float = 1e-4, seed: int = 42):
        self.k = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.rng = np.random.default_rng(seed)
        self.centroids: np.ndarray = np.array([])
        
    def fit(self, X: np.ndarray) -> "PureNumPyKMeans":
        """Fits cluster centroids to data matrix X of shape (N, D)."""
        n_samples, n_features = X.shape
        
        # 1. Randomly initialize centroids from sample points
        initial_indices = self.rng.choice(n_samples, size=self.k, replace=False)
        self.centroids = X[initial_indices].copy()
        
        for iteration in range(self.max_iters):
            # 2. Vectorized pairwise Euclidean distance calculation via broadcasting
            # Shape: X[:, np.newaxis, :] is (N, 1, D); centroids[np.newaxis, :, :] is (1, K, D)
            # Differences: (N, K, D) -> sum squares along feature axis -> (N, K)
            diffs = X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]
            dist_sq = np.sum(diffs ** 2, axis=2)
            
            # 3. Assign labels to nearest centroid
            labels = np.argmin(dist_sq, axis=1)
            
            # 4. Update centroids by computing cluster means
            new_centroids = np.zeros_like(self.centroids)
            for cluster_idx in range(self.k):
                cluster_mask = (labels == cluster_idx)
                if np.any(cluster_mask):
                    new_centroids[cluster_idx] = np.mean(X[cluster_mask], axis=0)
                else:
                    # Handle empty cluster by re-sampling a random point
                    new_centroids[cluster_idx] = X[self.rng.choice(n_samples)]
            
            # 5. Check for centroid shift convergence
            shift = np.linalg.norm(self.centroids - new_centroids)
            self.centroids = new_centroids
            if shift < self.tol:
                print(f"Converged at iteration {iteration} with shift {shift:.6f}")
                break
                
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts closest cluster index for query samples."""
        diffs = X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]
        dist_sq = np.sum(diffs ** 2, axis=2)
        return np.argmin(dist_sq, axis=1)

# Execution test
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    c1 = rng.normal(loc=[0, 0], scale=0.5, size=(100, 2))
    c2 = rng.normal(loc=[5, 5], scale=0.5, size=(100, 2))
    c3 = rng.normal(loc=[0, 5], scale=0.5, size=(100, 2))
    X_synthetic = np.vstack([c1, c2, c3])
    
    model = PureNumPyKMeans(n_clusters=3)
    model.fit(X_synthetic)
    preds = model.predict(X_synthetic[:5])
    print(f"Predictions for first 5 samples: {preds}")
```

---

## Key Points & Pitfalls

### 1. Accidental Memory Copies
- Basic slicing (`arr[0:5, :]`) creates a **view** (fast, zero memory overhead).
- Boolean indexing (`arr[arr > 0]`) and fancy indexing (`arr[[0, 2, 4]]`) always create a **copy**.
- If you modify a view, the original array changes; if you modify a copy, the original array remains untouched.

### 2. The `axis` Argument Demystified
The `axis` parameter specifies the dimension along which an operation collapses or operates:
- `axis=0`: Collapses the row axis (operates down each column), yielding one summary stat per column.
- `axis=1`: Collapses the column axis (operates across each row), yielding one summary stat per row.
- Remember: *Axis is the index of the dimension that disappears*.

### 3. Floating Point Precision & Numerical Stability
- Default floating-point dtype is `float64` (double precision). For deep learning, `float32` (single precision) halves memory usage compared to `float64`, and `float16`/`bfloat16`
  halve it again while matching GPU tensor core hardware capabilities.
- When computing log probabilities or softmax, subtract the maximum value first to prevent exponential overflow:
  $$\text{Softmax}(\mathbf{z}) = \frac{\exp(\mathbf{z} - \max(\mathbf{z}))}{\sum \exp(\mathbf{z} - \max(\mathbf{z}))}$$

---
