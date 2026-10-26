# Model Evaluation, Validation Protocols & Performance Metrics

## Concept
A machine learning model is only as credible as its validation protocol. The ultimate objective of model development
is not achieving high performance on observed training data, but guaranteeing **generalization** to unseen data
under real-world operational distributions.

Flawed evaluation protocols produce disastrous failures in production: data leakage gives false confidence during development,
aggregate accuracy conceals complete failure on minority classes, and improper validation splits result in immediate model decay.
Rigorous evaluation requires selecting task-appropriate loss metrics and implementing leak-proof cross-validation schemes.

---

## Formula / Core Idea

### 1. Classification Metrics & The Confusion Matrix

```text
                       Predicted Positive (1)     Predicted Negative (0)
Actual Positive (1)    True Positive (TP)         False Negative (FN) [Type II Error]
Actual Negative (0)    False Positive (FP)        True Negative (TN)  [Type I Error]
                       [Type I Error]
```

Core Derived Ratios:
- **Accuracy**: $\frac{\text{TP} + \text{TN}}{\text{Total}}$ (Deceptive under class imbalance).
- **Precision**: $\frac{\text{TP}}{\text{TP} + \text{FP}}$ (Of all positive predictions, how many were correct?).
- **Recall (Sensitivity)**: $\frac{\text{TP}}{\text{TP} + \text{FN}}$ (Of all actual positive cases, how many did we catch?).
- **Specificity**: $\frac{\text{TN}}{\text{TN} + \text{FP}}$ (True negative rate).
- **$F_1$-Score**: Harmonic mean of Precision and Recall:
  $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}$$

### 2. ROC-AUC vs PR-AUC
- **ROC Curve**: Plots True Positive Rate (Recall) vs False Positive Rate ($\frac{\text{FP}}{\text{FP}+\text{TN}}$) across all thresholds.
- **PR Curve**: Plots Precision vs Recall across all thresholds.
**Crucial Rule**: Use PR-AUC whenever evaluating models on rare event detection (e.g., fraud, cancer, defect detection).

### 3. Regression Error Metrics
- **Mean Squared Error (MSE)**: $\frac{1}{N} \sum (y_i - \hat{y}_i)^2$ (Penalizes large outlier errors heavily).
- **Root Mean Squared Error (RMSE)**: $\sqrt{\text{MSE}}$ (In the same physical units as the target variable).
- **Mean Absolute Error (MAE)**: $\frac{1}{N} \sum |y_i - \hat{y}_i|$ (Robust to outliers; median estimator).
- **Coefficient of Determination ($R^2$)**: $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ (Fraction of variance explained).

---

## When to Use

- **Stratified K-Fold**: Mandatory for imbalanced classification tasks to ensure each fold preserves class proportions.
- **TimeSeriesSplit**: Mandatory for temporal data; never train on future data to predict the past.
- **GroupKFold**: Mandatory when multiple observations originate from the same patient/user to prevent cross-split identity leakage.

---

## Example

Below is a pure Python implementation of comprehensive classification metrics without any external library dependencies:

```python
from typing import Dict
import numpy as np

def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates fundamental binary classification metrics from scratch."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "specificity": specificity,
        "confusion_matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
    }

if __name__ == "__main__":
    actuals = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    preds   = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    
    report = evaluate_classification(actuals, preds)
    print(f"Accuracy:   {report['accuracy']:.2f}")
    print(f"Precision:  {report['precision']:.2f}")
    print(f"Recall:     {report['recall']:.2f}")
    print(f"F1 Score:   {report['f1_score']:.2f}")
    print("Confusion:", report["confusion_matrix"])
```

---

## Key Points & Pitfalls

### 1. Data Leakage via Global Transformations
Calling `scaler.fit_transform(X)` on the complete dataset prior to cross-validation leaks test fold mean and variance
into the training fold. Always fit transformers **exclusively on the training partition**.

### 2. Time-Travel Leakage in Time-Series
Standard random shuffling on time-series data destroys temporal causality. Always use forward-chaining `TimeSeriesSplit`.

### 3. Evaluating on Train Set Only
High training accuracy combined with low validation accuracy indicates classic high variance (overfitting).

---

