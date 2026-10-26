# Multi-Layer Perceptrons (MLPs) & Activation Functions

## Concept
The Multi-Layer Perceptron (MLP) is the quintessential deep learning architecture. While a single artificial neuron
(the Perceptron) can only separate linearly separable classes, stacking layers of artificial neurons separated by
**non-linear activation functions** enables deep neural networks to approximate arbitrarily complex non-linear functions.

This capability is formalized by the **Universal Approximation Theorem** (Cybenko, 1989), which states that a feed-forward
network with a single hidden layer containing a finite number of neurons can approximate any continuous function on compact subsets
of $\mathbb{R}^D$ to arbitrary precision, provided the activation function is continuous and non-linear.

---

## Formula / Core Idea

### 1. Mathematical Anatomy of an MLP Layer
For layer $l \in \{1, \dots, L\}$ with input vector $\mathbf{a}^{l-1} \in \mathbb{R}^{D_{l-1}}$:
$$\mathbf{z}^l = W^l \mathbf{a}^{l-1} + \mathbf{b}^l$$
$$\mathbf{a}^l = \sigma(\mathbf{z}^l)$$

### 2. Comprehensive Activation Function Zoo
- **Sigmoid**: $\frac{1}{1 + e^{-z}} \in (0, 1)$. Vanishing gradients; outputs not zero-centered.
- **Tanh**: $\frac{e^z - e^{-z}}{e^z + e^{-z}} \in (-1, 1)$. Zero-centered; still suffers from vanishing gradients.
- **ReLU**: $\max(0, z)$. Fast computation; solves vanishing gradients; "Dying ReLU" issue.
- **Leaky ReLU**: $\max(\alpha z, z)$ with $\alpha \approx 0.01$. Prevents dead neurons.
- **GELU**: $z \Phi(z)$. Smooth non-linearity; standard in BERT, GPT, and modern Transformers.
- **Swish / SiLU**: $z \sigma(z)$. Smooth, non-monotonic; standard in Llama / Mistral.

### 3. Weight Initialization Protocols
- **Xavier (Glorot) Initialization** (For Sigmoid and Tanh activations): $W \sim \mathcal{N}\left(0, \frac{2}{D_{\text{in}} + D_{\text{out}}}\right)$.
- **He (Kaiming) Initialization** (For ReLU and Leaky ReLU activations): $W \sim \mathcal{N}\left(0, \frac{2}{D_{\text{in}}}\right)$.

---

## When to Use

- Use **MLPs** for dense tabular representations, final classification projection heads in CNNs and Transformers,
  and actor-critic networks in Reinforcement Learning.
- Default to **GELU** or **Swish** for Transformer architectures.
- Default to **ReLU** with **He Initialization** for standard convolutional and feed-forward networks.

---

## Example

Below is a complete PyTorch implementation of a modern Multi-Layer Perceptron with LayerNorm and GELU:

```python
import torch
import torch.nn as nn

class ModernMLP(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, out_features: int, dropout_rate: float = 0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, out_features)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

if __name__ == "__main__":
    model = ModernMLP(in_features=20, hidden_dim=64, out_features=2)
    sample_x = torch.randn(8, 20)
    logits = model(sample_x)
    print("Logits shape:", logits.shape)
```

---

## Key Points & Pitfalls
- Always initialize weights with He or Xavier scaling to prevent signal decay.
- Remember to call `model.eval()` before running inference with Dropout layers.
- Stacking linear layers without non-linearities collapses into a single linear regression.

---