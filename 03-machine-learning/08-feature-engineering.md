# Feature Engineering

## Concept
Creating and transforming input variables so models learn better.

## Formula / Core Idea
Scaling, encoding categories, handling missing values, creating new features, selecting features.

## When to Use
Before training, especially on tabular data.

## Example
```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
```

## Key Points
- Fit scalers on training data only (avoid leakage)
- One-hot encode low-cardinality categories
- Log transform skewed values
- Good features often beat fancier models

## My Notes
- (add your own notes here)
