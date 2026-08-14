# SVM and KNN

## Concept
SVM finds the maximum-margin boundary. KNN classifies by the nearest neighbours.

## Formula / Core Idea
SVM: maximise margin, kernels handle non-linear data. KNN: majority vote of k closest points.

## When to Use
Small to medium datasets, clear class separation.

## Example
```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
```

## Key Points
- Both need feature scaling
- KNN is slow on large data
- Choose k with cross-validation
- RBF kernel is a common SVM default

## My Notes
- (add your own notes here)
