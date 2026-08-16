# Model Evaluation

## Concept
Measuring how well a model will perform on unseen data.

## Formula / Core Idea
Accuracy, Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = 2PR/(P+R). Regression: MAE, RMSE, R-squared.

## When to Use
After every training run, and when comparing models.

## Example
```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

## Key Points
- Accuracy misleads on imbalanced data
- Use k-fold cross-validation
- Confusion matrix shows error types
- Never tune on the test set

## My Notes
- (add your own notes here)
