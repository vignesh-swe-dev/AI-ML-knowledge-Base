# Logistic Regression

## Concept
A classification model that outputs the probability of a class.

## Formula / Core Idea
p = 1 / (1 + e^-(w.x + b)). Loss: log loss (cross entropy).

## When to Use
Binary classification such as spam or not spam, churn or stay.

## Example
```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression().fit(X_train, y_train)
print(clf.predict_proba(X_test)[:5])
```

## Key Points
- Despite the name, it is a classifier
- Threshold of 0.5 can be tuned
- Scale features first
- Use precision, recall and ROC-AUC on imbalanced data

## My Notes
- (add your own notes here)
