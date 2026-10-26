# Module 01: Python Basics & The Scientific Stack

## 📌 Module Overview
Python is the core programming language for Artificial Intelligence, Machine Learning, and Scientific Computing.
Its widespread dominance in AI stems from two distinct factors:
1. **High Expressiveness & Ergonomics**: Rapid prototyping, dynamic duck-typing, and clean human-readable syntax.
2. **Native C/C++ Interoperability**: Seamless integration with low-level accelerated hardware libraries
   (BLAS, LAPACK, CUDA, OneMKL) through vectorized C extensions like NumPy.

This module provides a rigorous, ground-up foundation in the scientific computing stack.
Rather than treating libraries as black boxes, this module breaks down memory models, striding,
vectorized execution, and production software patterns.

---

## 📚 Syllabus & Navigation

| File | Topic | Primary Focus | Practical Application |
|---|---|---|---|
| [01-python-fundamentals.md](01-python-fundamentals.md) | Python Language Core | Data structures, OOP, decorators, generators | Production code quality & data pipelines |
| [02-numpy.md](02-numpy.md) | Array Computing | Vectorization, broadcasting, memory strides | High-speed linear algebra & numerical math |
| [03-pandas.md](03-pandas.md) | Tabular Data Systems | DataFrames, split-apply-combine, joins | Feature engineering & ETL preprocessing |
| [04-matplotlib.md](04-matplotlib.md) | Diagnostic Visualizations | Object-oriented API, custom styling, subplots | EDA, residual analysis, model evaluation |

---

## 🏗️ The Scientific Python Stack Architecture

Modern machine learning rests on a multi-tiered software hierarchy:

```text
+-------------------------------------------------------------------------------+
|                           High-Level ML & Deep Learning                        |
|             PyTorch · Scikit-Learn · TensorFlow · Hugging Face Transformers   |
+---------------------------------------+---------------------------------------+
|      Tabular & Feature Processing     |          Visualization Layer          |
|            Pandas · Polars            |          Matplotlib · Seaborn         |
+---------------------------------------+---------------------------------------+
|                         Array & Math Foundations                              |
|                             NumPy · SciPy                                     |
+-------------------------------------------------------------------------------+
|                    Low-Level Hardware & Computation Kernels                   |
|               OpenBLAS · Intel MKL · LAPACK · CUDA / cuBLAS · C / C++         |
+-------------------------------------------------------------------------------+
```

Understanding this hierarchy clarifies where computational bottlenecks occur and how to eliminate them.

---

## ⚙️ Environment Setup & Reproducibility

Professional ML development demands reproducible, deterministic environments. Relying on global
system-wide Python installations leads to package version conflicts and unreproducible experiments.

### Step 1: Create an Isolated Virtual Environment
```bash
# Create an isolated virtual environment
python -m venv .venv

# Activate on Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Activate on Linux / macOS
source .venv/bin/activate

# Update pip and build dependencies
python -m pip install --upgrade pip setuptools wheel
```

### Step 2: Production Dependency Pinning (`requirements.txt`)
```text
numpy>=1.26.0,<2.0.0
pandas>=2.1.0,<3.0.0
scipy>=1.11.0,<2.0.0
matplotlib>=3.8.0,<4.0.0
seaborn>=0.13.0,<1.0.0
scikit-learn>=1.3.0,<2.0.0
jupyterlab>=4.0.0,<5.0.0
ipywidgets>=8.1.0
pytest>=7.4.0
ruff>=0.1.5
mypy>=1.6.0
```

Install the entire environment via:
```bash
pip install -r requirements.txt
```

---

## ⏱️ Performance Paradigm: Vectorized vs Scalar Execution

The core axiom of numerical Python is: **Never use explicit loops over numerical arrays**.

### Why Python Loops Are Slow
1. **Dynamic Type Checking**: At every single iteration, CPython checks the type of the operands.
2. **Boxing & Unboxing Overhead**: Integers and floats in Python are heap-allocated `PyObject` structs with pointer overhead.
3. **Poor Cache Locality**: Python lists store references (pointers) to objects scattered across memory, causing cache misses.

### Why NumPy Is Fast
1. **Homogeneous Contiguous Memory**: Elements of a single type (`float64`) sit consecutively in RAM.
2. **C-Level Vectorization**: Inner loops execute compiled C/Fortran routines directly on CPU registers.
3. **SIMD Hardware Acceleration**: Modern CPUs execute Single Instruction Multiple Data instructions (AVX-512, NEON),
   computing 4 to 8 floating-point operations in a single clock cycle.

### Benchmark Demonstration
```python
import time
import numpy as np

SIZE = 10_000_000
py_list = list(range(SIZE))
np_array = np.arange(SIZE, dtype=np.float64)

# 1. Python List Comprehension
t_start = time.perf_counter()
py_res = [x * 2.5 + 1.5 for x in py_list]
t_py = time.perf_counter() - t_start

# 2. Vectorized NumPy Array Operation
t_start = time.perf_counter()
np_res = np_array * 2.5 + 1.5
t_np = time.perf_counter() - t_start

print(f"Python List loop:  {t_py:.4f} seconds")
print(f"NumPy Vectorized:  {t_np:.4f} seconds")
print(f"Speedup Factor:    {t_py / t_np:.1f}x faster")
```

Benchmark results typically demonstrate a **50x to 100x speedup** with a fraction of the RAM overhead.

---

## 🛠️ Code Quality & Static Analysis

Maintainable machine learning pipelines adhere to strict typing and automated linting standards.

### 1. Static Type Checking with Mypy
Type hints eliminate shape and type misunderstandings across data transformations:
```python
from typing import Sequence, Tuple
import numpy as np
from numpy.typing import NDArray

def standardize_features(
    features: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Standardizes 2D feature matrix to zero mean and unit variance."""
    mean: NDArray[np.float64] = np.mean(features, axis=0)
    std: NDArray[np.float64] = np.std(features, axis=0)
    # Avoid division by zero for invariant features
    std_safe = np.where(std == 0.0, 1.0, std)
    standardized = (features - mean) / std_safe
    return standardized, mean, std
```

### 2. High-Speed Linting with Ruff
Run Ruff to automatically detect anti-patterns, unused imports, and style violations:
```bash
ruff check . --fix
```

---

## 🧪 Self-Assessment & Hands-On Exercises

Test your mastery of Module 01 fundamentals by completing these practical coding challenges:

1. **Custom Batch Generator with Reshuffle**:
   Implement a generator class in Python that yields mini-batches of size $B$ from features $X$ and labels $y$,
   shuffling sample order at the beginning of each epoch without copying the underlying arrays.

2. **Vectorized Mahalanobis Distance**:
   Given a data matrix $X \in \mathbb{R}^{N \times D}$ and covariance matrix $\Sigma \in \mathbb{R}^{D \times D}$,
   compute the pairwise Mahalanobis distance between all points and the centroid without using any Python `for` loops.

3. **Memory-Optimized Parquet ETL**:
   Load a multi-million row CSV dataset into Pandas, downcast `float64` columns to `float32`,
   convert low-cardinality string columns to `category` dtype, and export to snappy-compressed Parquet.
   Quantify the memory footprint reduction.

4. **Diagnostic Residual Visualizer**:
   Using Matplotlib's Object-Oriented API (`fig, ax = plt.subplots(2, 2)`), build a reusable 4-panel diagnostic chart:
   Residuals vs Fitted, Normal Q-Q Plot, Scale-Location Plot, and Cook's Distance Plot.

---

## 💡 Top 5 Technical Interview Questions (Python for ML)

### Q1: What is the Python GIL and why doesn't it slow down NumPy matrix multiplications?
**Answer**: The Global Interpreter Lock (GIL) is a mutual exclusion lock in CPython ensuring thread safety by
allowing only one thread to execute Python bytecode at a time. However, NumPy C extensions release the GIL
prior to entering compiled BLAS/LAPACK routines. Multi-threaded operations such as matrix multiplications (`@`)
execute across all physical CPU cores using OpenMP or Intel MKL independent of the GIL.

### Q2: How does Python's memory model handle variable assignment and parameter passing?
**Answer**: Python uses "pass-by-object-reference" (or pass-by-assignment). Variables are pointers (references)
bound to objects in memory. For immutable objects (ints, strings, tuples), operations return new objects.
For mutable objects (lists, dicts, NumPy arrays), modifying the object in-place inside a function alters
the caller's data buffer unless an explicit copy was created.

### Q3: What is the difference between shallow copy and deep copy?
**Answer**: A shallow copy (`copy.copy()` or `arr.view()`) creates a new container object but inserts references
to the child objects found in the original. A deep copy (`copy.deepcopy()` or `arr.copy()`) recursively
duplicates all nested objects, guaranteeing complete independence of the memory buffers.

### Q4: When should you use a Python Generator instead of a List Comprehension?
**Answer**: Use list comprehensions when you require random access, slicing, or need to iterate multiple times
over small-to-medium collections. Use generators (using `yield` or generator expressions `(x for x in data)`)
when processing streaming data, infinite sequences, or massive datasets that cannot fit simultaneously in RAM.

### Q5: What is Method Chaining in Pandas and what are its pros and cons?
**Answer**: Method chaining (`df.pipe().assign().query()`) chains DataFrame transformations into a continuous,
declarative pipeline without creating temporary variables. Pros: Clean, readable ETL pipeline flow matching functional
paradigms. Cons: Debugging intermediate steps can be harder without inserting `.pipe(lambda d: ...)` loggers.

---

## 📝 Study Checklist & Milestones
- [ ] Complete Python fundamentals review (OOP, closures, decorators, generators).
- [ ] Master NumPy array creation, striding, indexing, and multi-axis reductions.
- [ ] Practice broadcasting rules until two-dimensional and three-dimensional array alignment is intuitive.
- [ ] Build robust Pandas preprocessing pipelines using `groupby()`, `merge()`, and categorical downcasting.
- [ ] Create publication-grade exploratory plots using Matplotlib Object-Oriented figures.
