# Quick Reference: Scikit-Learn API & Production Workflows

## 1. Unified Estimator API Pattern
Every algorithm in Scikit-Learn implements a standardized object-oriented interface:
```python
from sklearn.base import BaseEstimator, ClassifierMixin

# 1. Instantiation with Hyperparameters
# model = ModelClass(param1=val1, param2=val2)

# 2. Fitting to Training Data (Learns internal state)
# model.fit(X_train, y_train)

# 3. Transforming Data (Transformers only)
# X_transformed = transformer.transform(X_test)

# 4. Making Predictions (Estimators only)
# y_pred = model.predict(X_test)
# y_probs = model.predict_proba(X_test)  # Returns probabilities (N, K)
```

## 2. Leak-Proof Preprocessing with `ColumnTransformer`
```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier

num_cols = ['tenure', 'monthly_charges']
cat_cols = ['contract', 'payment_method']

# Define specialized sub-pipelines
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine into ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
])

# Assemble full end-to-end Pipeline
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])
```

## 3. Cross-Validation & Hyperparameter Search
```python
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import uniform, randint

param_dist = {
    'classifier__learning_rate': uniform(0.01, 0.2),
    'classifier__max_iter': randint(50, 200),
    'classifier__max_leaf_nodes': randint(15, 63)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    estimator=full_pipeline,
    param_distributions=param_dist,
    n_iter=20,
    scoring='roc_auc',
    cv=cv,
    random_state=42,
    n_jobs=-1
)

# Preprocessing is re-fit inside each training fold, preventing leakage across folds
search.fit(X_train, y_train)
print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
best_model = search.best_estimator_
```

## 4. Model Persistence & Serialization
```python
import joblib

# Save trained pipeline to disk
joblib.dump(best_model, 'champion_pipeline.joblib')

# Load in production inference service
loaded_pipeline = joblib.load('champion_pipeline.joblib')
predictions = loaded_pipeline.predict(X_new)
```
