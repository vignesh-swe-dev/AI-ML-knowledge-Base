# Decision Trees & Random Forests: Ensembles & CART

## Concept
Decision Trees are intuitive, non-parametric supervised learning models that partition the input feature space
into a hierarchical sequence of axis-aligned orthogonal rectangular regions. For any query sample,
the model traverses root-to-leaf decision nodes, returning the majority class (classification) or mean target value (regression).

While single decision trees suffer from high variance (they are notorious overfitters prone to capturing data noise),
combining hundreds of decorrelated decision trees via **Bagging (Bootstrap Aggregation)** and random feature subsampling
creates the **Random Forest** algorithm (Breiman, 2001). Random Forests consistently rank among the most accurate,
robust, and widely deployed algorithms for tabular machine learning.

---

## Formula / Core Idea

### 1. Splitting Criteria (CART Algorithm)
At each node $m$, the algorithm searches over all available features $j$ and candidate thresholds $t$ to find
the split $(j, t)$ that maximizes the impurity reduction:
$$\Delta I(m) = I(m) - \left( \frac{N_L}{N_m} I(L) + \frac{N_R}{N_m} I(R) \right)$$

Where impurity $I(m)$ is quantified by:
- **Gini Impurity (Default in Scikit-Learn)**:
  $$I_{\text{Gini}}(m) = 1 - \sum_{k=1}^K p_{mk}^2$$
- **Shannon Entropy (Information Gain)**:
  $$I_{\text{Entropy}}(m) = - \sum_{k=1}^K p_{mk} \log_2(p_{mk})$$
- **Variance Reduction (Regression Trees)**:
  $$I_{\text{MSE}}(m) = \frac{1}{N_m} \sum_{i \in m} (y_i - \bar{y}_m)^2$$

### 2. Random Forest Variance Reduction Mathematics
Suppose we ensemble $B$ identically distributed trees, each with variance $\sigma^2$ and pairwise correlation $\rho$.
The variance of the average ensemble prediction $\bar{f} = \frac{1}{B} \sum_{b=1}^B T_b$ is:
$$\text{Var}(\bar{f}) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$

As the number of trees $B \to \infty$, the second term $\frac{1-\rho}{B}\sigma^2 \to 0$.
However, the ensemble variance cannot fall below $\rho \sigma^2$.
**Random Forests actively minimize $\rho$** by forcing each tree split to consider only a random subset of features
(typically $\sqrt{D}$ for classification and $D/3$ for regression), ensuring the trees are strongly decorrelated!

---

## When to Use

- **Tabular Datasets with Non-Linear Interactions**: When complex feature interactions exist without requiring manual polynomial feature engineering.
- **Heterogeneous Features**: Trees do not require feature scaling or normalization. (In Scikit-Learn, categorical features still need to be encoded as numbers.)
- **Feature Importance Auditing**: When stakeholders require global feature ranking via Permutation Importance or Gini drop.

---

## Example

Below is a complete, self-contained Python script training a Random Forest with Out-of-Bag (OOB) scoring and permutation feature importance:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np

# 1. Load benchmark dataset
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 2. Train Random Forest with OOB Evaluation enabled
rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=6,
    max_features="sqrt",       # Decorrelates trees
    oob_score=True,             # Approximate cross-validation estimate
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

print(f"Out-of-Bag (OOB) Generalization Score: {rf.oob_score_:.4f}")
print(f"Holdout Test Set Accuracy:            {rf.score(X_test, y_test):.4f}")

# 3. Compute Permutation Feature Importance
perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
top_indices = np.argsort(perm_importance.importances_mean)[::-1][:5]

print("\nTop 5 Most Informative Features (Permutation Importance):")
for idx in top_indices:
    name = feature_names[idx]
    score = perm_importance.importances_mean[idx]
    print(f"- {name:<28}: {score:+.4f}")
```

---

## Key Points & Pitfalls

### 1. Inherent Bias of Gini Impurity Importance
Mean Decrease in Impurity (MDI, `rf.feature_importances_`) is heavily biased toward high-cardinality features
(e.g. numerical IDs or timestamps) because they offer more split points. Prefer **Permutation Feature Importance**
(`permutation_importance`, computed on held-out data) for feature auditing, keeping in mind that it can understate the importance of strongly correlated features.

### 2. Extrapolation Blind Spots
Tree-based models **cannot extrapolate** outside the numerical minimum and maximum values seen during training.
For time-series forecasting with a continuous trend, detrend the target first or use models that can extrapolate (e.g., linear models or neural networks).

### 3. Tree Pruning Parameters
To prevent individual trees from memorizing training noise, constrain tree capacity using:
- `max_depth`: Limits maximum distance from root to leaf.
- `min_samples_split`: Minimum instances required to attempt a split.
- `min_samples_leaf`: Minimum instances required to exist in any terminal leaf node.

---
