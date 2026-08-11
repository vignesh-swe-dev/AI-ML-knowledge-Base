# Linear Regression

## Concept
Predicts a continuous value using a straight-line relationship between inputs and output.

## Formula / Core Idea
y = w1*x1 + w2*x2 + ... + b. Loss: Mean Squared Error = (1/n) * sum((y - y_pred)^2).

## When to Use
Price, sales, temperature prediction where the relationship is roughly linear.

## Example
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X_train, y_train)
print(model.score(X_test, y_test))
```

## Key Points
- Assumes linearity and independent errors
- Sensitive to outliers
- Use Ridge or Lasso to reduce overfitting
- Evaluate with RMSE, MAE and R-squared

## My Notes
- (add your own notes here)
