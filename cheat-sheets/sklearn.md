# Scikit-learn Cheat Sheet

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
print(cross_val_score(pipe, X, y, cv=5).mean())
```

Pattern: `fit()` to learn, `predict()` to infer, `score()` to evaluate.
