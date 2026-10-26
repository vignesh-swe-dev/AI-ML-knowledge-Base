# Backpropagation, Computational Graphs & Modern Optimizers

## Concept
Backpropagation is the computational workhorse of deep learning. It is an efficient implementation of
the multivariate calculus chain rule applied to directed acyclic **computational graphs**.

Training a deep neural network requires calculating the partial derivative of the scalar loss $\mathcal{L}$
with respect to every weight parameter in the network: $\frac{\partial \mathcal{L}}{\partial W^l}$.
While forward-mode automatic differentiation would require $O(\text{Parameters})$ passes through the network,
**Reverse-Mode Automatic Differentiation** (Backpropagation) computes the exact gradient with respect to *all* parameters
in a single backward sweep, requiring computational time proportional to only a constant multiple of the forward pass.

---

## Formula / Core Idea

### 1. Reverse-Mode Autodiff Derivation
Consider a neuron in layer $l$ computing affine pre-activation $z^l = w^l a^{l-1} + b^l$ and activation $a^l = \sigma(z^l)$.
Define the error sensitivity (adjoint) at layer $l$ as:
$$\delta^l \equiv \frac{\partial \mathcal{L}}{\partial \mathbf{z}^l}$$

Using the multivariate chain rule:
1. **Output Layer Error ($l = L$)**:
   $$\delta^L = \nabla_{\mathbf{a}^L} \mathcal{L} \odot \sigma'(\mathbf{z}^L)$$
2. **Backward Recurrence Relation ($l < L$)**:
   $$\delta^l = \left( (W^{l+1})^T \delta^{l+1} \right) \odot \sigma'(\mathbf{z}^l)$$
3. **Weight Gradients**:
   $$\frac{\partial \mathcal{L}}{\partial W^l} = \delta^l (\mathbf{a}^{l-1})^T$$
4. **Bias Gradients**:
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^l} = \delta^l$$

### 2. State-of-the-Art Optimizers Comparison

| Optimizer | Mathematical Update Rule | Primary Strength |
|---|---|---|
| **SGD + Momentum** | $\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \mathbf{g}_t$; $\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \mathbf{v}_t$ | Escapes shallow ravines; excellent test generalization |
| **RMSprop** | $\mathbf{s}_t = \alpha \mathbf{s}_{t-1} + (1-\alpha)\mathbf{g}_t^2$; $\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{\mathbf{s}_t}+\epsilon}\mathbf{g}_t$ | Normalizes step size across varying gradient scales |
| **Adam** | Combines 1st moment $\mathbf{m}_t$ (momentum) & 2nd moment $\mathbf{v}_t$ (RMSprop) | Fast initial convergence; standard baseline |
| **AdamW** | Decouples weight decay from gradient moment accumulation | State-of-the-art for Transformers & LLMs |

---

## When to Use

- Use **AdamW** with cosine decay learning rate schedule as the default choice for Transformers, LLMs, and Diffusion models.
- Use **SGD with Nesterov Momentum** for training state-of-the-art Computer Vision networks (ResNets, ConvNeXt).
- Implement **Gradient Clipping** when training recurrent models or deep transformers to prevent numerical explosion.

---

## Example

Below is a complete, self-contained implementation of a 2-layer neural network with pure NumPy backpropagation:

```python
import numpy as np

class PureNumPyTwoLayerNet:
    """Two-layer MLP demonstrating explicit forward and backward backpropagation passes."""
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, lr: float = 0.05):
        self.lr = lr
        # He initialization for ReLU hidden layer
        self.W1 = np.random.randn(hidden_dim, in_dim) * np.sqrt(2.0 / in_dim)
        self.b1 = np.zeros((hidden_dim, 1))
        # Xavier initialization for output layer
        self.W2 = np.random.randn(out_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + out_dim))
        self.b2 = np.zeros((out_dim, 1))
        
    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)
        
    @staticmethod
    def relu_grad(z: np.ndarray) -> np.ndarray:
        return (z > 0).astype(float)
        
    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        """Executes single batch forward pass, computes MSE loss, and executes backpropagation."""
        N = X.shape[1]  # Samples along columns
        
        # 1. Forward Pass
        z1 = self.W1 @ X + self.b1
        a1 = self.relu(z1)
        z2 = self.W2 @ a1 + self.b2
        y_hat = z2
        
        # 2. Loss (MSE)
        loss = (1.0 / (2.0 * N)) * np.sum((y_hat - y) ** 2)
        
        # 3. Backward Pass (Backpropagation)
        delta2 = (1.0 / N) * (y_hat - y)                   # Output adjoint
        grad_W2 = delta2 @ a1.T
        grad_b2 = np.sum(delta2, axis=1, keepdims=True)
        
        delta1 = (self.W2.T @ delta2) * self.relu_grad(z1) # Backpropagated hidden adjoint
        grad_W1 = delta1 @ X.T
        grad_b1 = np.sum(delta1, axis=1, keepdims=True)
        
        # 4. Gradient Descent Parameter Updates
        self.W2 -= self.lr * grad_W2
        self.b2 -= self.lr * grad_b2
        self.W1 -= self.lr * grad_W1
        self.b1 -= self.lr * grad_b1
        
        return float(loss)

if __name__ == "__main__":
    np.random.seed(42)
    X_train = np.random.randn(4, 100) # 4 features, 100 samples
    y_train = np.sin(X_train[:1, :])  # Non-linear target
    
    net = PureNumPyTwoLayerNet(in_dim=4, hidden_dim=16, out_dim=1, lr=0.1)
    for epoch in range(100):
        l = net.train_step(X_train, y_train)
        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch+1:03d} | MSE Loss: {l:.6f}")
```

---

## Key Points & Pitfalls

### 1. Vanishing vs Exploding Gradients
- Vanishing: Adjoint $\delta^l \to 0$. Solved by skip connections, normalization, and ReLU/GELU activations.
- Exploding: Adjoint $\delta^l \to \infty$, leading to NaN losses. Solved by Gradient Norm Clipping.

### 2. Forgetting `optimizer.zero_grad()`
In PyTorch, gradients accumulate by default (`grad += new_grad`) to facilitate multi-step batch accumulation.
Forgetting to call `optimizer.zero_grad()` results in corrupt, ever-growing gradients.

### 3. Gradient Checking (Finite Differences)
When debugging custom backward operations, verify analytical gradients against numerical approximations:
$$\frac{\partial f}{\partial \theta} \approx \frac{f(\theta + \epsilon) - f(\theta - \epsilon)}{2\epsilon}$$

---

