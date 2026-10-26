# Python Fundamentals for Machine Learning Engineering

## Concept
Python is the foundational programming language of the modern artificial intelligence ecosystem.
While high-level frameworks like PyTorch and Scikit-Learn provide turnkey mathematical abstractions,
robust, production-grade model development requires a deep understanding of Python's execution model,
reference semantics, memory management, and advanced language constructs.

In machine learning systems, inefficient Python code frequently causes GPU under-utilization (GPU starvation)
because data loading and preprocessing in Python worker processes cannot keep pace with tensor accelerators.
Mastery of generators, iterators, closures, decorators, and object-oriented abstractions ensures
data ingestion pipelines remain clean, memory-efficient, and highly scalable.

---

## Formula / Core Idea

### 1. Python Memory Model & Reference Counting
Every variable in Python is a reference (pointer) to an object allocated on the heap.
CPython manages memory via reference counting supplemented by a cyclic garbage collector.

```text
Variable Name (Namespace Stack)         Heap Memory Object
+-----------------------------+         +-------------------------------+
|              a              | ------> | ob_refcnt = 2                 |
+-----------------------------+         | ob_type   = PyList_Type       |
                                        | ob_size   = 3                 |
+-----------------------------+         | ob_item   = [ptr1, ptr2, ptr3]|
|              b              | ------> +-------------------------------+
+-----------------------------+
```

When variable `b = a`, no data is copied. Both variables point to the exact same `PyObject`.
For mutable containers (lists, dicts, custom objects), mutating `b` immediately modifies `a`.

### 2. Time & Space Complexity of Core Data Structures

| Data Structure | Operation | Average Complexity | Worst Case | Memory Layout |
|---|---|---|---|---|
| **List** | Index lookup `arr[i]` | $O(1)$ | $O(1)$ | Dynamic array of pointers |
| **List** | Append `arr.append(x)` | $O(1)$ amortized | $O(N)$ (during resize) | Contiguous pointer buffer |
| **List** | Insert/Delete `arr.insert(0, x)` | $O(N)$ | $O(N)$ | Requires shifting $N$ pointers |
| **Deque** | Append/Pop Left | $O(1)$ | $O(1)$ | Doubly linked list of blocks |
| **Dict** | Lookup / Insert / Delete | $O(1)$ | $O(N)$ (hash collisions) | Compact hash table (sparse array + dense entries) |
| **Set** | Member check `x in s` | $O(1)$ | $O(N)$ | Hash table (keys only) |

### 3. Generator Expressions & The Iterator Protocol
The iterator protocol in Python requires two dunder methods:
1. `__iter__()`: Returns the iterator object itself.
2. `__next__()`: Returns the next element in the sequence, raising `StopIteration` when exhausted.

Generators simplify this protocol using `yield`, maintaining execution frame state while consuming $O(1)$ memory:
$$\text{Memory}(\text{List Comprehension}) = O(N) \quad \text{vs.} \quad \text{Memory}(\text{Generator}) = O(1)$$

---

## When to Use

### When to Use Functional Patterns (Comprehensions & Generators)
- **List Comprehensions**: Use when data fits comfortably in RAM and you need random indexing or multiple iterations.
- **Generators & Iterators**: Use when streaming unbounded data, loading large batches of image paths,
  or reading gigabyte-scale CSVs line-by-line to prevent out-of-memory (OOM) crashes.
- **Closures & Decorators**: Use for cross-cutting concerns: profiling training latency, caching intermediate embeddings,
  or validating input tensor shapes before model forward passes.

### When to Use Object-Oriented Patterns (OOP)
- Creating custom Scikit-Learn transformers (`BaseEstimator`, `TransformerMixin`).
- Implementing PyTorch dataset loaders (`torch.utils.data.Dataset`).
- Encapsulating stateful model pipelines (hyperparameters, fitted weights, preprocessing states).

---

## Example

Below is a complete, production-grade implementation of a stateful dataset preprocessor and streaming batcher:

```python
from typing import Generator, List, Dict, Any, Tuple, Optional
import math
import time
from functools import wraps

# 1. Production Timing Decorator
def benchmark(func):
    """Decorator to benchmark function execution latency."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        print(f"[{func.__name__}] executed in {elapsed:.6f} seconds")
        return result
    return wrapper

# 2. Custom Dataset Loader with Iterator Protocol
class MLDataset:
    """A clean, memory-efficient Dataset class supporting streaming mini-batches."""
    
    def __init__(self, data: List[Dict[str, Any]], target_key: str):
        self._data = data
        self._target_key = target_key
        self._mean: Optional[float] = None
        self._std: Optional[float] = None
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._data[idx]
    
    @benchmark
    def fit_normalizer(self, feature_key: str) -> None:
        """Computes mean and standard deviation for a continuous feature."""
        values = [sample[feature_key] for sample in self._data if feature_key in sample]
        if not values:
            raise ValueError(f"Feature {feature_key} not present in dataset.")
        self._mean = sum(values) / len(values)
        variance = sum((x - self._mean) ** 2 for x in values) / len(values)
        self._std = math.sqrt(variance) if variance > 0 else 1.0
        print(f"Fitted normalizer: mean={self._mean:.4f}, std={self._std:.4f}")
    
    def batch_stream(
        self, batch_size: int
    ) -> Generator[Tuple[List[float], List[Any]], None, None]:
        """Yields mini-batches of normalized features and targets lazily."""
        if self._mean is None or self._std is None:
            raise RuntimeError("Normalizer must be fitted before batch streaming.")
            
        for i in range(0, len(self._data), batch_size):
            batch_samples = self._data[i : i + batch_size]
            batch_x = [(s["feature"] - self._mean) / self._std for s in batch_samples]
            batch_y = [s[self._target_key] for s in batch_samples]
            yield batch_x, batch_y

# Demonstration execution
if __name__ == "__main__":
    raw_records = [{"feature": float(i), "label": i % 2} for i in range(100)]
    dataset = MLDataset(raw_records, target_key="label")
    dataset.fit_normalizer("feature")
    
    stream = dataset.batch_stream(batch_size=32)
    for batch_idx, (X_batch, y_batch) in enumerate(stream):
        print(f"Batch {batch_idx}: {len(X_batch)} samples | First normalized val: {X_batch[0]:.4f}")
```

---

## Key Points & Pitfalls

### 1. The Mutable Default Argument Bug
**Problem**: Default argument expressions in Python functions are evaluated once when the function definition is executed,
not each time the function is called.
```python
# BAD: The list is shared across all function calls!
def append_embedding(embedding: list, history: list = []):
    history.append(embedding)
    return history

# GOOD: Use None as default sentinel
def append_embedding(embedding: list, history: Optional[list] = None):
    if history is None:
        history = []
    history.append(embedding)
    return history
```

### 2. Shallow Copy vs Deep Copy in Weights & Configs
Mutating an inner list or array inside a shallow copy inadvertently corrupts original model configurations.
Always use `copy.deepcopy()` when cloning nested hyperparameters or model state dictionaries.

### 3. Generator Exhaustion
Generators are single-pass consumers. Once iterated through, they are permanently exhausted.
Attempting a second loop over an exhausted generator silently yields 0 items without throwing an error.
To reuse data across multiple epochs, instantiate a fresh generator or use a class implementing `__iter__`.

### 4. Overuse of `*args` and `**kwargs`
While flexible, excessive use of variable keyword arguments obscures function signatures, breaks IDE autocompletion,
and prevents static type checkers (`mypy`) from detecting configuration typos in model architectures.

---
