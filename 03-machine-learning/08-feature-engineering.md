# Feature Engineering, Encoding & Transformation Pipelines

## Concept
Feature engineering is the process of transforming raw domain data into structured, predictive numerical representations
that machine learning algorithms can effectively exploit. As Turing Award laureate Andrew Ng famously stated:
*"Coming up with features is difficult, time-consuming, requires expert knowledge. 'Applied machine learning' is basically feature engineering."*

Raw real-world datasets arrive with missing observations, high-cardinality categorical variables, skewed distributions,
and redundant collinear predictors. A well-designed feature engineering pipeline stabilizes numerical optimization,
improves linear separability, and prevents catastrophic data leakage.

---

## Formula / Core Idea

### 1. Numerical Feature Transformations
- **Standardization ($Z$-Score Scaling)**: Centers data to zero mean and unit variance:
  $$z = \frac{x - \mu}{\sigma}$$
  Mandatory for gradient descent, SVM, KNN, and Ridge/Lasso regularization.
- **Min-Max Normalization**: Bounds features into the fixed range $[0, 1]$:
  $$x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$
- **Power / Box-Cox Transformation**: Stabilizes variance and maps skewed distributions to approximate Gaussians:
  $$y^{(\lambda)} = \begin{cases} \frac{y^\lambda - 1}{\lambda} & \text{if } \lambda \ne 0 \\ \ln(y) & \text{if } \lambda = 0 \end{cases}$$

### 2. Categorical Encoding Strategies

| Method | Description | Cardinality Suitability | Pros & Cons |
|---|---|---|---|
| **One-Hot Encoding** | Creates $C$ binary columns | Low ($C < 20$) | Avoids artificial ordering; causes dimensionality explosion |
| **Ordinal Encoding** | Maps categories to integers $0, 1, \dots, C-1$ | Ordered categories | Preserves natural ranking (e.g. Low, Med, High) |
| **Target Encoding** | Replaces category with expected target mean: $\mathbb{E}[y \mid x = c]$ | High cardinality | Extremely predictive; severe risk of target leakage without smoothing |

---

## When to Use

- **One-Hot Encoding**: Unordered low-cardinality nominal categories (e.g., Blood Type, Department).
- **Target Encoding**: High-cardinality nominal variables (e.g., Zip Codes, Device IDs, City).
- **RobustScaler**: When datasets contain heavy-tailed outliers that distort mean and standard deviation.

---

## Example

Below is a complete, production-grade Scikit-Learn `ColumnTransformer` pipeline preprocessing mixed tabular data:

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import numpy as np

# 1. Create sample heterogeneous DataFrame
df_sample = pd.DataFrame({
    "age": [25.0, np.nan, 45.0, 32.0, 58.0],
    "income": [50000.0, 75000.0, 120000.0, np.nan, 95000.0],
    "education": ["Bachelors", "Masters", "PhD", "Bachelors", "HighSchool"],
    "city": ["NewYork", "London", "NewYork", "Tokyo", "London"],
    "purchased": [0, 1, 1, 0, 1]
})

numeric_features = ["age", "income"]
categorical_features = ["education", "city"]

# 2. Define modular pipelines per data type
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

# 3. Combine pipelines via ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# 4. Fit and transform training data
processed_features = preprocessor.fit_transform(df_sample)
print(f"Original Shape:  {df_sample.shape}")
print(f"Processed Shape: {processed_features.shape}")
print("Transformed Array:\n", np.round(processed_features[:2], 3))
```

---

## Key Points & Pitfalls

### 1. Target Leakage in Target Encoding
Replacing categories with in-sample target averages memorizes labels, leading to extreme overfitting.
Always use **Out-of-Fold (OOF)** target encoding or apply empirical Bayes smoothing with a global prior.

### 2. Unseen Categories at Inference Time
In production, test samples often contain new categories unseen during training.
Always configure `OneHotEncoder(handle_unknown='ignore')` to output all zeros for unseen categories rather than throwing an exception.

### 3. Log Transforms on Zero or Negative Values
Applying $\log(x)$ to features containing zero throws an error or outputs $-\infty$.
Use $\log(1 + x)$ via `np.log1p()` or Yeo-Johnson transformation to safely handle zero values.

---

ncy, and verify reproducibility with fixed random seeds.
