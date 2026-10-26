# Pandas: Tabular Data Systems & Feature Engineering

## Concept
Pandas is the premier open-source data manipulation and analysis library for Python.
While NumPy provides the low-level numerical ndarray foundation, real-world data science datasets
are heterogeneous, featuring mixed types (integers, floats, categorical labels, strings, timestamps),
missing values (NaN / None), and arbitrary row and column labels.

Pandas provides two primary data structures: the 1D labeled **Series** and the 2D tabular **DataFrame**.
Its internal architecture relies on the **BlockManager** (or the modern Arrow backend), grouping homogeneous
columns together in contiguous memory to enable vectorized operations while supporting rich relational
operations: joining, merging, grouping, reshaping, and time-series rolling windows.

---

## Formula / Core Idea

### 1. Data Structure Architecture
- **Series**: A 1D array with an explicit index. Internally holds a NumPy array or Arrow chunked array.
- **DataFrame**: A collection of Series aligned to a shared Index. Columns can have distinct data types.

```text
DataFrame: Shape (4, 3)
Index        'Age' (int32)    'Salary' (float64)    'Department' (category)
  0      ->       25                 50000.0                'Sales'
  1      ->       32                 75000.0                'Engineering'
  2      ->       45                 92000.0                'Engineering'
  3      ->       29                 61000.0                'Marketing'
```

### 2. The Split-Apply-Combine Pattern
The standard analytical workflow for aggregated statistics follows Hadley Wickham's pattern:
1. **Split**: Partition the DataFrame into disjoint subsets based on one or more categorical keys.
2. **Apply**: Compute a summary reduction (mean, sum), transformation (z-score), or filtration on each group.
3. **Combine**: Assemble individual group outputs back into a unified DataFrame or Series.

$$\mu_g = \frac{1}{|S_g|} \sum_{i \in S_g} x_i, \quad \forall g \in \text{Groups}$$

### 3. Memory Optimization: Downcasting & Categorical Types
By default, Pandas imports text columns as `object` (arrays of pointers to Python strings) and numeric columns as `int64` or `float64`.
This creates massive RAM overhead. For a column with repeated string labels:
- `object` dtype: Stores an 8-byte pointer per row + individual `PyUnicode` heap allocations.
- `category` dtype: Stores an integer code lookup table (e.g. `int8` for up to 127 categories), which often reduces memory by well over **80%** for low-cardinality columns.

---

## When to Use

- **Exploratory Data Analysis (EDA)**: Profiling missing value rates, data distributions, and feature correlations.
- **Data Cleaning & Wrangling**: Imputing null values, removing duplicates, clipping statistical outliers.
- **Feature Engineering**: Creating interaction features, rolling moving averages, date-time decompositions.
- **Data Ingestion & Serialization**: Reading and writing CSV, Parquet, Feather, JSON, and SQL database tables.

---

## Example

Below is an end-to-end, production-grade feature engineering pipeline for tabular customer churn data:

```python
import pandas as pd
import numpy as np

# 1. Create representative raw DataFrame
np.random.seed(42)
data = {
    "customer_id": [f"CUST_{i:04d}" for i in range(1000)],
    "contract_type": np.random.choice(["Month-to-Month", "One-Year", "Two-Year"], size=1000),
    "tenure_months": np.random.randint(1, 72, size=1000),
    "monthly_charges": np.random.uniform(20.0, 120.0, size=1000),
    "total_charges": np.random.uniform(100.0, 8000.0, size=1000),
    "churn": np.random.choice([0, 1], p=[0.75, 0.25], size=1000),
}
df_raw = pd.DataFrame(data)
# Introduce realistic missing values
df_raw.loc[::50, "total_charges"] = np.nan

# 2. Functional Preprocessing & Feature Engineering Pipeline
def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Downcasts numeric types and converts strings to categorical."""
    df = df.copy()
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["object"]).columns:
        if col != "customer_id":
            df[col] = df[col].astype("category")
    return df

def clean_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Imputes missing total_charges using tenure * monthly_charges."""
    df = df.copy()
    imputed_val = df["tenure_months"] * df["monthly_charges"]
    df["total_charges"] = df["total_charges"].fillna(imputed_val)
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates derived analytical domain features."""
    return df.assign(
        charges_per_tenure=lambda d: d["total_charges"] / np.maximum(d["tenure_months"], 1),
        high_value_customer=lambda d: (d["monthly_charges"] > d["monthly_charges"].quantile(0.80)).astype(int),
    )

# 3. Execute declarative pipeline using method chaining
df_processed = (
    df_raw
    .pipe(optimize_memory)
    .pipe(clean_missing)
    .pipe(engineer_features)
)

# 4. Split-Apply-Combine summary
summary = (
    df_processed
    .groupby("contract_type", observed=False)
    .agg(
        sample_count=("customer_id", "count"),
        mean_monthly=("monthly_charges", "mean"),
        churn_rate=("churn", "mean"),
    )
    .reset_index()
)

print("Processed DataFrame Head:")
print(df_processed.head(3))
print("\nContract Summary:")
print(summary)
```

---

## Key Points & Pitfalls

### 1. `SettingWithCopyWarning`
This warning occurs when performing chained indexing: `df['col'][mask] = value`.
Pandas cannot guarantee whether the first index returned a view or a copy, resulting in ambiguous updates.
**Fix**: Always use explicit `.loc` indexing:
```python
# CORRECT:
df.loc[mask, 'col'] = value
```

### 2. Avoid Iterating with `iterrows()`
`df.iterrows()` converts each row into a Pandas Series at every iteration, destroying type consistency and running
far slower (often by orders of magnitude) than vectorized expressions and noticeably slower than `itertuples()`. Avoid it for large tables.

### 3. Merging vs Concatenation
- `pd.merge()`: Performs SQL-style joins (`inner`, `left`, `right`, `outer`) on common keys or indexes.
- `pd.concat()`: Stacks DataFrames along an axis (row-wise `axis=0` or column-wise `axis=1`).

---
