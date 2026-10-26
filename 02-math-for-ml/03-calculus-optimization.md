# Calculus & Optimization for Machine Learning

## Concept
If linear algebra represents data and probability models uncertainty, multivariate calculus and optimization
govern how machine learning models actually learn. Training a parameterized model $f_\theta(\mathbf{x})$
is fundamentally the process of finding parameter vector $\theta^*$ that minimizes an empirical loss function $\mathcal{L}(\theta)$.

Because closed-form analytic solutions are impossible for non-linear models and deep neural networks,
machine learning relies on iterative, gradient-based numerical optimization. Understanding multivariate derivatives,
Jacobians, Hessians, and first-order optimization dynamics is essential for stabilizing training runs,
preventing gradient explosion/vanishing, and selecting learning rate schedules.

---

## Formula / Core Idea

### 1. Vector Calculus Foundations
- **Gradient Vector $\nabla f(\mathbf{w})$**: For scalar function $f: \mathbb{R}^D \to \mathbb{R}$, the gradient is the vector of all partial derivatives:
  $$\nabla f(\mathbf{w}) = \left[ \frac{\partial f}{\partial w_1}, \frac{\partial f}{\partial w_2}, \dots, \frac{\partial f}{\partial w_D} \right]^T \in \mathbb{R}^D$$
  The gradient points in the direction of **steepest ascent**. Negative gradient $-\nabla f$ points toward steepest descent.

- **Jacobian Matrix $J$**: For vector-valued function $\mathbf{f}: \mathbb{R}^D \to \mathbb{R}^M$:
  $$J_{ij} = \frac{\partial f_i}{\partial w_j} \in \mathbb{R}^{M \times D}$$

- **Hessian Matrix $H$**: For scalar function $f: \mathbb{R}^D \to \mathbb{R}$, the matrix of second-order partial derivatives:
  $$H_{ij} = \frac{\partial^2 f}{\partial w_i \partial w_j} \in \mathbb{R}^{D \times D}$$
  If $H$ is Positive Definite ($H \succ 0$, all eigenvalues $\lambda_i > 0$), the stationary point is a strict local minimum.

### 2. Gradient Descent Variants
The general parameter update rule at step $t$ is:
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \cdot \mathbf{g}_t$$
where $\eta > 0$ is the learning rate, and $\mathbf{g}_t$ is the gradient estimate.

| Optimizer | Update Equations | Key Mechanism |
|---|---|---|
| **Vanilla SGD** | $\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla \mathcal{L}(\mathbf{w}_t)$ | Direct step along current batch gradient |
| **Polyak Momentum** | $\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \nabla \mathcal{L}(\mathbf{w}_t)$<br>$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \mathbf{v}_t$ | Dampens oscillations in high-curvature directions |
| **RMSprop** | $\mathbf{s}_t = \gamma \mathbf{s}_{t-1} + (1-\gamma) \mathbf{g}_t^2$<br>$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{\mathbf{s}_t} + \epsilon} \mathbf{g}_t$ | Normalizes step size by historical gradient magnitudes |
| **Adam (Adaptive Moments)** | $\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\mathbf{g}_t$<br>$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)\mathbf{g}_t^2$<br>$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \hat{\mathbf{m}}_t$ | Combines momentum with per-parameter adaptive scales |

---

## When to Use

- **Vanilla SGD with Momentum**: Frequently preferred for computer vision (ResNets, ConvNets) where it often achieves better generalization.
- **Adam & AdamW**: Default industry standard for Transformers, Large Language Models (LLMs), and Reinforcement Learning.
- **Learning Rate Schedulers**: Cosine annealing with warm restarts to escape sharp local minima and settle into flat minima.

---

## Example

Below is a complete implementation comparing **Vanilla Gradient Descent** vs **Adam Optimizer** on the challenging 2D Rosenbrock loss surface:

```python
import numpy as np

def rosenbrock(x: np.ndarray) -> float:
    """Classic non-convex benchmark: f(x, y) = (a - x)^2 + b(y - x^2)^2 with a=1, b=100."""
    return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2

def rosenbrock_grad(x: np.ndarray) -> np.ndarray:
    """Analytic gradient vector [df/dx, df/dy]^T."""
    df_dx = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] ** 2)
    df_dy = 200.0 * (x[1] - x[0] ** 2)
    return np.array([df_dx, df_dy])

# 1. Pure NumPy Adam Optimizer
class PureNumPyAdam:
    def __init__(self, lr: float = 0.05, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(2)
        self.v = np.zeros(2)
        self.t = 0
        
    def step(self, w: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        # Exponential moving averages of gradients and squared gradients
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad ** 2)
        # Bias corrections for initial zero-bias
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)
        # Parameter update
        return w - (self.lr / (np.sqrt(v_hat) + self.eps)) * m_hat

# Execution test
if __name__ == "__main__":
    init_pos = np.array([-1.5, 2.0])  # Start far from minimum (1, 1)
    
    # Test Adam Optimizer
    w_adam = init_pos.copy()
    opt = PureNumPyAdam(lr=0.02)
    for step in range(500):
        grad = rosenbrock_grad(w_adam)
        w_adam = opt.step(w_adam, grad)
        
    final_loss = rosenbrock(w_adam)
    print("Target Global Minimum: [1.0, 1.0]")
    print(f"Adam Position after 500 steps: [{w_adam[0]:.4f}, {w_adam[1]:.4f}]")
    print(f"Final Rosenbrock Loss: {final_loss:.6f}")
```

---

## Key Points & Pitfalls

### 1. Learning Rate Selection ($O(\eta)$)
- Too small $\eta$: Crawling convergence, prone to getting trapped in high saddle points.
- Too large $\eta$: Divergence and NaN loss values due to overshoot over steep ravine walls.

### 2. Saddle Points vs Local Minima
In high-dimensional spaces ($D > 10^6$), local minima with high error almost never exist.
The primary impediments to optimization are **saddle points** where the gradient vanishes ($\nabla f = \mathbf{0}$)
but the Hessian has both positive and negative eigenvalues. Momentum enables escape along negative curvature directions.

### 3. Gradient Explosion & Vanishing
In deep networks, backpropagation chains multiple Jacobian multiplications: $\prod_{l=1}^L J_l$.
If the largest singular values $\sigma(J) > 1$, gradients explode exponentially (mitigate with Gradient Clipping: $\|\mathbf{g}\| \le c$).
If $\sigma(J) < 1$, gradients vanish exponentially (mitigate with Residual Skip Connections and Layer Normalization).

---

