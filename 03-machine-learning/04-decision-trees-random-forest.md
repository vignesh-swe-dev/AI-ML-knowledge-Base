# Decision Trees and Random Forest

## Concept
Trees split data using feature questions. Random Forest averages many trees.

## Formula / Core Idea
Split quality via Gini impurity or entropy. Forest = bagging + random feature subsets.

## When to Use
Tabular data, mixed feature types, a strong baseline.

## Example
```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200).fit(X_train, y_train)
```

## Key Points
- Single trees overfit easily
- Forests reduce variance
- Feature importance is available
- Gradient boosting (XGBoost, LightGBM) is often even stronger

## My Notes
- (add your own notes here)
