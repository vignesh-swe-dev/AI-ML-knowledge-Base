# Linear Regression: OLS, Regularization & Ridge / Lasso

## Concept
Linear regression is the foundational supervised learning method for modeling the quantitative relationship
between a continuous scalar target variable $y \in \mathbb{R}$ and one or more explanatory features $\mathbf{x} \in \mathbb{R}^D$.

The linear model expresses the target prediction $\hat{y}$ as an affine combination of input features:
$$\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_D x_D = \mathbf{w}^T \mathbf{x}$$

Despite its structural simplicity, linear regression remains immensely valuable in machine learning due to its mathematical
tractability, explicit interpretability (each weight represents the expected change in target per unit change in feature),
and existence of a closed-form global optimum via Ordinary Least Squares (OLS). When augmented with $L_1$ and $L_2$
regularization (Ridge, Lasso, ElasticNet), linear regression remains competitive on high-dimensional tabular datasets.

---

## Formula / Core Idea

### 1. Ordinary Least Squares (OLS) Closed-Form Derivation
Given design matrix $X \in \mathbb{R}^{N \times (D+1)}$ and target vector $\mathbf{y} \in \mathbb{R}^N$:
$$\text{Loss}(\mathbf{w}) = \frac{1}{2N} \|X \mathbf{w} - \mathbf{y}\|_2^2$$
$$\nabla_\mathbf{w} \text{Loss}(\mathbf{w}) = \frac{1}{N} X^T (X \mathbf{w} - \mathbf{y}) = \mathbf{0}$$
$$X^T X \mathbf{w} = X^T \mathbf{y} \implies \mathbf{w}^* = (X^T X)^{-1} X^T \mathbf{y}$$

### 2. Regularized Loss Formulations
- **Ridge ($L_2$)**: $\frac{1}{2N} \|X\mathbf{w} - \mathbf{y}\|^2 + \lambda \|\mathbf{w}\|_2^2 \implies \mathbf{w}^* = (X^T X + \lambda I)^{-1} X^T \mathbf{y}$.
- **Lasso ($L_1$)**: $\frac{1}{2N} \|X\mathbf{w} - \mathbf{y}\|^2 + \alpha \|\mathbf{w}\|_1$. Enforces sparsity and feature selection.
- **ElasticNet**: Combines $L_1$ and $L_2$ penalties for handling groups of correlated features.

---

## When to Use

- **Econometric & Financial Modeling**: When exact interpretability and statistical significance are required.
- **High-Dimensional Datasets ($D \gg N$)**: Ridge and Lasso prevent severe overfitting.

---

## Example

Below is a complete implementation of **Ridge Regression** from first-principles using NumPy:

```python
import numpy as np

class PureNumPyRidgeRegression:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.weights_: np.ndarray = np.array([])
        self.intercept_: float = 0.0
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> "PureNumPyRidgeRegression":
        N, D = X.shape
        X_bias = np.hstack([np.ones((N, 1)), X])
        reg_matrix = np.eye(D + 1)
        reg_matrix[0, 0] = 0.0
        A = X_bias.T @ X_bias + self.alpha * reg_matrix
        b = X_bias.T @ y
        w_full = np.linalg.solve(A, b)
        self.intercept_ = float(w_full[0])
        self.weights_ = w_full[1:]
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights_ + self.intercept_

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    X_synth = rng.normal(size=(200, 5))
    true_w = np.array([3.5, -2.0, 0.0, 0.5, 1.2])
    true_b = 4.2
    y_synth = X_synth @ true_w + true_b + rng.normal(scale=0.5, size=200)
    
    model = PureNumPyRidgeRegression(alpha=0.5)
    model.fit(X_synth, y_synth)
    print(f"Fitted Intercept: {model.intercept_:.4f}")
    print("Fitted Weights:", np.round(model.weights_, 4))
```

---

## Key Points & Pitfalls
- If features are collinear, $X^T X$ is singular; Ridge guarantees invertibility.
- Standardize features before applying regularization.
- Never penalize the intercept/bias term.

---

