# Pandas

## Concept
Pandas handles tabular data using DataFrame and Series.

## Formula / Core Idea
Read, clean, filter, group and merge data.

## When to Use
Data exploration and preprocessing before modelling.

## Example
```python
import pandas as pd
df = pd.read_csv('data.csv')
print(df.head())
print(df.isna().sum())
print(df.groupby('city')['price'].mean())
```

## Key Points
- Check `df.info()` and `df.describe()` first
- Handle missing values deliberately
- Use `groupby`, `merge`, `pivot_table`
- Avoid chained assignment

## My Notes
- (add your own notes here)
