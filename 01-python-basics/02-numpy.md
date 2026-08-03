# NumPy

## Concept
NumPy provides fast n-dimensional arrays and vectorised maths.

## Formula / Core Idea
Arrays replace Python loops. Operations are applied element-wise.

## When to Use
Any numeric computation, matrices, ML maths.

## Example
```python
import numpy as np
a = np.array([[1,2],[3,4]])
print(a.T)
print(a @ a)
print(a.mean(axis=0))
```

## Key Points
- Shape, dtype and axis are the key concepts
- Broadcasting lets different shapes combine
- Vectorise instead of looping
- `@` is matrix multiplication

## My Notes
- (add your own notes here)
