# Large Language Models: Architectures, Decoding & LoRA

## Concept
Large Language Models (LLMs) are scaled, autoregressive Transformer models trained on massive internet-scale text corpora.
Formulated mathematically as conditional probability estimators, LLMs model the joint probability of a sequence of tokens
via the chain rule of probability:
$$P(w_1, w_2, \dots, w_T) = \prod_{t=1}^T P(w_t \mid w_1, \dots, w_{t-1})$$

Through massive compute scale (scaling laws: Kaplan et al., 2020; Chinchilla: Hoffmann et al., 2022),
these autoregressive models transition from simple statistical next-token predictors to general-purpose reasoning engines
capable of few-shot learning, tool use, code generation, and complex multi-turn dialogue.

---

## Formula / Core Idea

### 1. Autoregressive Decoding Dynamics
At step $t$, the decoder generates unnormalized log-probabilities (logits) $\mathbf{z} \in \mathbb{R}^{|V|}$ across vocabulary $V$.
To control output creativity and randomness, logits are scaled by **Temperature $T > 0$**:
$$P(w_t = i) = \frac{\exp(z_i / T)}{\sum_{j \in V} \exp(z_j / T)}$$

- $T \to 0$: Approaches greedy search (argmax), selecting the highest-probability token.
- $T = 1$: Unmodified model probability distribution.
- $T > 1$: Flattens the distribution, increasing token diversity and hallucinations.

### 2. Top-$P$ (Nucleus) Sampling Mathematical Formulation
Sort vocabulary in descending order of probability: $p_1 \ge p_2 \ge \dots \ge p_{|V|}$.
Find the smallest index $k$ such that:
$$\sum_{i=1}^k p_i \ge P$$
Discard all tokens with index $> k$, re-normalize the remaining probability mass to sum to 1, and sample from this truncated nucleus.

### 3. Parameter-Efficient Fine-Tuning: LoRA
Instead of updating pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA decomposes the update into low-rank matrices:
$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$

```text
     Original Weight W_0 (Frozen)                 LoRA Adapter Path (Trainable)
        d x k Matrix                              d x r Matrix B     r x k Matrix A
       +------------+                               +-------+         +-------------+
       |            |                               |       |    *    |             |
       |            |                               +-------+         +-------------+
       +------------+                                (Init: 0)         (Init: Gauss)
             │                                           │                   │
             └───────────────────────( + )◄──────────────┴───────────────────┘
                                       │
                                    Output
```

Where rank $r \ll \min(d, k)$, matrix $A$ is initialized from Gaussian $\mathcal{N}(0, \sigma^2)$, and $B$ is initialized to zero,
guaranteeing $\Delta W = 0$ at the start of fine-tuning.

---

## When to Use

- Use **Greedy Search ($T=0$)** for deterministic coding, mathematical reasoning, and factual question answering.
- Use **Top-$P$ ($P=0.9, T=0.7$)** for creative copywriting, conversational chatbots, and story generation.
- Use **LoRA / QLoRA** to adapt open-source foundation models (e.g. Llama 3 8B) on consumer GPUs with limited VRAM.

---

## Example

Below is a complete implementation of temperature scaling, top-$k$, and top-$p$ nucleus sampling from raw logits in NumPy:

```python
import numpy as np

def sample_next_token(logits: np.ndarray, temperature: float = 0.7, top_k: int = 50, top_p: float = 0.9) -> int:
    """Applies Temperature, Top-K, and Top-P filtering to sample the next token ID."""
    # 1. Apply temperature scaling
    if temperature <= 1e-4:
        return int(np.argmax(logits))
    logits = logits / temperature
    
    # 2. Convert to probabilities via Softmax with numerical stability
    logits_shifted = logits - np.max(logits)
    probs = np.exp(logits_shifted) / np.sum(np.exp(logits_shifted))
    
    # 3. Apply Top-K filtering
    if top_k > 0 and top_k < len(probs):
        k_indices = np.argsort(probs)[-top_k:]
        mask = np.zeros_like(probs, dtype=bool)
        mask[k_indices] = True
        probs[~mask] = 0.0
        probs = probs / np.sum(probs)
        
    # 4. Apply Top-P (Nucleus) filtering
    if top_p < 1.0:
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        
        # Identify cutoff index where cumulative mass exceeds top_p
        cutoff_index = np.searchsorted(cumulative_probs, top_p)
        valid_sorted_indices = sorted_indices[:cutoff_index + 1]
        
        filtered_probs = np.zeros_like(probs)
        filtered_probs[valid_sorted_indices] = probs[valid_sorted_indices]
        probs = filtered_probs / np.sum(filtered_probs)
        
    # 5. Sample token index from final categorical distribution
    return int(np.random.choice(len(probs), p=probs))

if __name__ == "__main__":
    np.random.seed(42)
    # Simulate raw output logits for a vocabulary of 1000 tokens
    simulated_logits = np.random.randn(1000)
    # Set a few tokens to have high scores
    simulated_logits[42] = 5.2
    simulated_logits[100] = 4.8
    simulated_logits[256] = 4.5
    
    sampled_id = sample_next_token(simulated_logits, temperature=0.7, top_k=10, top_p=0.85)
    print(f"Sampled Next Token ID: {sampled_id}")
```

---

## Key Points & Pitfalls

### 1. KV-Cache Memory Consumption
KV-Cache size scales linearly with context length $L$, batch size $B$, layers $N_{\text{layers}}$, and heads $N_{\text{heads}}$:
$$\text{Memory (bytes)} = 2 \times 2 \times B \times L \times N_{\text{layers}} \times d_{\text{model}}$$
For long context windows ($128k$ tokens), the KV-cache easily exceeds model parameter memory. Use Grouped-Query Attention (GQA)
to reduce KV-cache footprints by $8\times$.

### 2. Repetition & Degeneration
Greedy search frequently traps models in degenerate repetitive loops ("the the the").
Apply frequency penalties or use Nucleus sampling ($P=0.9$) to preserve linguistic diversity.

### 3. Quantization Degradation (FP16 $\to$ INT4)
4-bit quantization (bitsandbytes, AWQ, GPTQ) reduces VRAM by 75% with negligible perplexity loss, but can degrade code generation
and complex mathematical reasoning if outlier feature channels are clipped.

---