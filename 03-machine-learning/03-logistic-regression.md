# Logistic Regression & Multi-Class Softmax Classification

## Concept
Despite its name, Logistic Regression is a foundational linear model for **classification** rather than regression.
While linear regression models unbounded real-valued targets $\hat{y} \in (-\infty, \infty)$, classification requires
predicting the probability $P(Y = 1 \mid \mathbf{x}) \in [0, 1]$ that an input vector belongs to a positive class.

Logistic regression bridges this gap by mapping the linear combination of inputs $z = \mathbf{w}^T \mathbf{x} + b$ through
the non-linear **Sigmoid (logistic) link function**. In multi-class settings with $K > 2$ discrete categories,
this link function generalizes to the **Softmax function**, which parameterizes a categorical probability distribution.

---

## Formula / Core Idea

### 1. The Sigmoid Link & Odds Ratio
The Sigmoid function $\sigma(z)$ squashes any real scalar $z \in \mathbb{R}$ into the probability interval $(0, 1)$:
$$\sigma(z) = \frac{1}{1 + e^{-z}} = \frac{e^z}{1 + e^z}$$

Properties of Sigmoid:
- Derivative: $\frac{d\sigma}{dz} = \sigma(z)(1 - \sigma(z))$
- Symmetry: $\sigma(-z) = 1 - \sigma(z)$
- Log-odds (logit transformation):
  $$\log \left( \frac{P(Y=1\mid \mathbf{x})}{1 - P(Y=1\mid \mathbf{x})} \right) = \mathbf{w}^T \mathbf{x} + b$$
  This proves that the *log-odds of the positive class is a strictly linear function* of the input features.

### 2. Binary Cross-Entropy Loss (Log-Loss)
For $N$ samples with binary targets $y_i \in \{0, 1\}$ and predicted probabilities $\hat{y}_i = \sigma(\mathbf{w}^T \mathbf{x}_i + b)$:
$$\mathcal{L}(\mathbf{w}) = - \frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

The gradient of Binary Cross-Entropy with respect to weight vector $\mathbf{w}$ simplifies elegantly to:
$$\nabla_\mathbf{w} \mathcal{L} = \frac{1}{N} X^T (\hat{\mathbf{y}} - \mathbf{y})$$
Notice the striking mathematical similarity to OLS linear regression! The difference lies entirely in non-linear $\hat{\mathbf{y}}$.

### 3. Multi-Class Softmax Regression
For $K$ classes, we learn a weight vector $\mathbf{w}_k$ for each class:
$$P(Y = k \mid \mathbf{x}) = \frac{\exp(\mathbf{w}_k^T \mathbf{x})}{\sum_{j=1}^K \exp(\mathbf{w}_j^T \mathbf{x})}$$

---

## When to Use

- **Click-Through Rate (CTR) & Conversion Prediction**: In online advertising where well-calibrated probabilities are critical.
- **Credit Scoring & Risk Modeling**: In banking, where model decisions must be legally explainable via feature odds ratios.
- **High-Throughput Low-Latency Inference**: Logistic regression evaluates in microseconds, ideal for edge devices.

---

## Example

Below is a complete, vectorized implementation of **Logistic Regression** trained with Gradient Descent in NumPy:

```python
import numpy as np

class PureNumPyLogisticRegression:
    """Binary Logistic Regression classifier with Gradient Descent."""
    
    def __init__(self, lr: float = 0.1, n_iters: int = 1000, reg_lambda: float = 0.01):
        self.lr = lr
        self.n_iters = n_iters
        self.reg_lambda = reg_lambda
        self.weights: np.ndarray = np.array([])
        self.bias: float = 0.0
        
    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # Numerically stable sigmoid to avoid overflow
        return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> "PureNumPyLogisticRegression":
        N, D = X.shape
        self.weights = np.zeros(D)
        self.bias = 0.0
        
        for epoch in range(self.n_iters):
            # 1. Forward pass
            linear_logits = X @ self.weights + self.bias
            y_hat = self._sigmoid(linear_logits)
            
            # 2. Compute error
            errors = y_hat - y
            
            # 3. Compute gradients with L2 regularization
            grad_w = (1.0 / N) * (X.T @ errors) + (self.reg_lambda / N) * self.weights
            grad_b = (1.0 / N) * np.sum(errors)
            
            # 4. Parameter update
            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b
            
        return self
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._sigmoid(X @ self.weights + self.bias)
        
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

# Verification test
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    X_pos = rng.normal(loc=[2.0, 2.0], scale=1.0, size=(100, 2))
    X_neg = rng.normal(loc=[-2.0, -2.0], scale=1.0, size=(100, 2))
    X_test_data = np.vstack([X_pos, X_neg])
    y_test_data = np.array([1] * 100 + [0] * 100)
    
    clf = PureNumPyLogisticRegression(lr=0.5, n_iters=500)
    clf.fit(X_test_data, y_test_data)
    preds = clf.predict(X_test_data)
    accuracy = np.mean(preds == y_test_data)
    print(f"Training Accuracy: {accuracy * 100:.2f}%")
    print(f"Learned Decision Boundary: {clf.weights[0]:.2f}*x1 + {clf.weights[1]:.2f}*x2 + {clf.bias:.2f} = 0")
```

---

## Key Points & Pitfalls

### 1. Perfectly Separable Data & Infinite Weights
If the training data is linearly separable, the Maximum Likelihood weights will diverge to infinity ($\pm \infty$)
in an attempt to make $\sigma(z) \to 1.0$ and $\sigma(z) \to 0.0$. Regularization ($\lambda > 0$) is mandatory to stabilize weights.

### 2. Probability Calibration vs Classification Accuracy
A model can have 90% accuracy but horrific probability calibration (e.g. predicting 0.99 for instances that only occur 60% of the time).
Use Brier Score and Reliability Curves (`sklearn.calibration.CalibrationDisplay`) to evaluate calibration.

### 3. Decision Threshold Tuning
The default 0.5 classification threshold is arbitrary. If False Negatives are ten times costlier than False Positives,
lower the decision threshold (e.g. to 0.15) to boost recall.

---
