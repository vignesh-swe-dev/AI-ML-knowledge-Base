# Quick Reference: NumPy & Pandas Syntax

## 1. NumPy Array Creation & Inspection
```python
import numpy as np

# Creation
a = np.zeros((3, 4), dtype=np.float32)         # Zeros matrix
b = np.ones((2, 3), dtype=np.int64)            # Ones matrix
c = np.full((3, 3), fill_value=7.0)            # Constant matrix
d = np.eye(4)                                  # Identity matrix 4x4
e = np.arange(0, 10, step=2)                   # [0, 2, 4, 6, 8]
f = np.linspace(0.0, 1.0, num=50)              # 50 equidistant points in [0, 1]

# Random Number Generation (Modern API)
rng = np.random.default_rng(seed=42)
g = rng.normal(loc=0.0, scale=1.0, size=(100, 5)) # Gaussian Normal
h = rng.uniform(low=0.0, high=10.0, size=50)       # Uniform distribution
i = rng.choice([0, 1], size=100, p=[0.7, 0.3])     # Categorical sampling

# Inspection Attributes
a.shape   # Dimensions tuple (e.g. (3, 4))
a.ndim    # Number of dimensions (e.g. 2)
a.dtype   # Data type (e.g. float32)
a.strides # Byte step counts per dimension
```

## 2. NumPy Indexing, Slicing & Reshaping
```python
arr = np.arange(24).reshape(4, 6)

# Slicing (Returns a VIEW)
sub_view = arr[1:3, 2:5]     # Rows 1-2, Cols 2-4
row_vec  = arr[0, :]         # First row as 1D vector
col_vec  = arr[:, 2:3]       # Third column as 2D column matrix (4, 1)

# Advanced Indexing (Returns a COPY)
bool_mask = arr > 10
filtered  = arr[bool_mask]   # 1D array of elements matching condition
fancy_idx = arr[[0, 2], :]   # Rows 0 and 2

# Reshaping & Dimension Manipulation
flat     = arr.ravel()                 # 1D flattened view (if contiguous)
reshaped = arr.reshape(2, 3, 4)        # New shape with same total elements
expanded = arr[:, np.newaxis, :]       # Shape (4, 1, 6) for broadcasting
sqz      = np.squeeze(expanded)        # Drops dimensions of size 1 -> (4, 6)
```

## 3. NumPy Vectorized Math & Reductions
```python
A = rng.normal(size=(4, 4))
B = rng.normal(size=(4, 4))

# Linear Algebra
C = A @ B                        # Matrix multiplication (np.matmul)
dot_prod = np.dot(A[0], B[0])    # 1D inner product
A_inv = np.linalg.inv(A)         # Matrix inverse (use np.linalg.solve instead)
eigenvals, eigenvecs = np.linalg.eig(A)
U, S, Vt = np.linalg.svd(A)

# Multi-Axis Reductions
np.sum(A, axis=0)                # Sum across rows -> yields vector of col sums
np.mean(A, axis=1)               # Mean across cols -> yields vector of row means
np.std(A, axis=0, ddof=1)        # Sample standard deviation (divisor N - 1)
np.argmax(A, axis=1)             # Indices of maximum values along rows
```

## 4. Pandas DataFrame Indexing & Selection
```python
import pandas as pd

# Creation
df = pd.DataFrame({
    'age': [25, 30, 35, 40],
    'city': ['NYC', 'London', 'Tokyo', 'Paris'],
    'salary': [70000.0, 85000.0, 92000.0, 65000.0]
})

# loc (Label-based) vs iloc (Position-based)
val1 = df.loc[0, 'city']         # Row label 0, Column 'city'
val2 = df.iloc[0, 1]             # Row position 0, Column position 1
sub_df = df.loc[df['salary'] > 75000, ['age', 'city']] # Boolean query with column subset
```

## 5. Pandas Split-Apply-Combine & Merges
```python
# Groupby Aggregation
summary = df.groupby('city', as_index=False).agg(
    mean_salary=('salary', 'mean'),
    headcount=('age', 'count')
)

# Method Chaining & Assignment
df_clean = (
    df
    .query('age >= 21')
    .assign(
        tax=lambda d: d['salary'] * 0.25,
        is_high_earner=lambda d: d['salary'] > 80000
    )
    .sort_values(by='salary', ascending=False)
)

# SQL-Style Merging
merged = pd.merge(df1, df2, on='user_id', how='left')
stacked = pd.concat([df1, df2], axis=0, ignore_index=True)
```
