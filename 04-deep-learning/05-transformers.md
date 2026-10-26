# The Transformer Architecture & Self-Attention

## Concept
Introduced by Vaswani et al. in their landmark 2017 paper *"Attention Is All You Need"*, the Transformer architecture
completely revolutionized Artificial Intelligence, replacing recurrent architectures and becoming the universal foundation
for Large Language Models (LLMs), Vision Transformers (ViT), AlphaFold, and multi-modal generative models.

The core insight of the Transformer is eliminating sequential recurrent connections entirely in favor of
**Multi-Head Self-Attention**. Self-attention allows every token in a sequence to attend to and aggregate information
from every other token in a single constant-time ($O(1)$ path length) parallel operation, resolving the bottleneck of RNNs.

---

## Formula / Core Idea

### 1. Scaled Dot-Product Attention
Given Query matrix $Q \in \mathbb{R}^{N \times d_k}$, Key matrix $K \in \mathbb{R}^{M \times d_k}$, and Value matrix $V \in \mathbb{R}^{M \times d_v}$:
$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} + M \right) V$$

Where:
- $Q K^T$: Computes all pairwise token affinities (inner products).
- $\frac{1}{\sqrt{d_k}}$: Scaling factor preventing the dot products from growing excessively large for high dimensions,
  which would push the Softmax function into regions with near-zero gradients.
- $M$: Attention mask (e.g. causal triangular mask for autoregressive generation, setting future positions to $-\infty$).
- Softmax: Converts affinity scores into a normalized probability distribution across keys.

### 2. Multi-Head Attention (MHA)
Rather than performing a single attention function with $d_{\text{model}}$-dimensional queries, keys, and values,
Multi-Head Attention linearly projects $Q, K, V$ into $h$ distinct lower-dimensional subspaces of dimension $d_k = d_{\text{model}} / h$:
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$
$$\text{where} \quad \text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

### 3. Pre-LN Transformer Block Architecture
Modern LLMs (Llama, Mistral, GPT-NeoX) utilize the **Pre-LayerNorm** formulation for superior training stability:
$$\mathbf{x}^{(1)} = \mathbf{x} + \text{MultiHead}(\text{LN}(\mathbf{x}))$$
$$\mathbf{x}^{(2)} = \mathbf{x}^{(1)} + \text{FFN}(\text{LN}(\mathbf{x}^{(1)}))$$

Where $\text{FFN}(\mathbf{z}) = \text{GELU}(\mathbf{z} W_1 + \mathbf{b}_1) W_2 + \mathbf{b}_2$.

---

## When to Use

- **Autoregressive Text Generation**: Decoder-only Transformers (GPT-4, Llama 3) for text synthesis and instruction following.
- **Bidirectional Language Understanding**: Encoder-only Transformers (BERT, RoBERTa) for classification and embeddings.
- **Computer Vision**: Vision Transformers (ViT) dividing images into $16 \times 16$ pixel patches.

---

## Example

Below is a clean, modular PyTorch implementation of **Scaled Dot-Product Multi-Head Self-Attention**:

```python
import torch
import torch.nn as nn
import math

class MultiHeadSelfAttention(nn.Module):
    """Vectorized Multi-Head Self-Attention module."""
    
    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Projections for Q, K, V in a single fused linear layer
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, N, D = x.shape  # Batch size, Seq length, d_model
        
        # 1. Project and split into Q, K, V
        qkv = self.qkv_proj(x) # Shape: (B, N, 3 * d_model)
        qkv = qkv.reshape(B, N, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # Each shape: (B, n_heads, N, d_k)
        
        # 2. Scaled Dot-Product Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
            
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 3. Context aggregation and output projection
        context = torch.matmul(attn_weights, v) # Shape: (B, n_heads, N, d_k)
        context = context.permute(0, 2, 1, 3).reshape(B, N, D)
        return self.out_proj(context)

if __name__ == "__main__":
    mha = MultiHeadSelfAttention(d_model=256, n_heads=8)
    sample_seq = torch.randn(2, 64, 256) # Batch=2, Seq_Len=64, Dim=256
    out = mha(sample_seq)
    print("Input Sequence Shape: ", sample_seq.shape)
    print("Attended Output Shape:", out.shape)
```

---

## Key Points & Pitfalls

### 1. The Scaling Factor $\frac{1}{\sqrt{d_k}}$
Without scaling by $\sqrt{d_k}$, the variance of $q \cdot k$ equals $d_k$. For large dimensions ($d_k = 128$), dot products
explode to large values, pushing the Softmax function into saturated regions with near-zero gradients.

### 2. Causal Masking in Generative Models
In autoregressive generation, token $t$ must never attend to future tokens $t+1, t+2$.
Apply an upper-triangular mask with $-\infty$ above the diagonal prior to softmax.

### 3. Quadratic Complexity Mitigation
Standard self-attention is $O(N^2)$ in time and memory. Use **FlashAttention-2** (kernel fusion in SRAM) to train
on sequences beyond $32k$ tokens without OOM errors.

---
