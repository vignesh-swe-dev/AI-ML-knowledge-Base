# Recurrent Neural Networks (RNNs) & Long Short-Term Memory (LSTM)

## Concept
Standard feed-forward neural networks assume all inputs and outputs are independent and identically distributed (i.i.d.).
This assumption breaks down when modeling sequential data where the meaning of a current element depends strictly on context
from preceding elements, such as natural language sentences, financial time-series, and biosignals (ECG).

Recurrent Neural Networks (RNNs) address this by introducing **cycles in the computational graph**, maintaining
an internal **hidden state vector** $\mathbf{h}_t$ that acts as a recurrent memory. However, basic vanilla RNNs
suffer from severe vanishing gradients over long horizons. The **Long Short-Term Memory (LSTM)** architecture (Hochreiter & Schmidhuber, 1997)
solved this by introducing a linear cell state highway regulated by multiplicative gating mechanisms.

---

## Formula / Core Idea

### 1. Vanilla RNN Mechanics
At time step $t$ with input $\mathbf{x}_t$ and previous hidden state $\mathbf{h}_{t-1}$:
$$\mathbf{h}_t = \tanh(W_{hh} \mathbf{h}_{t-1} + W_{xh} \mathbf{x}_t + \mathbf{b}_h)$$
$$\hat{\mathbf{y}}_t = \text{softmax}(W_{hy} \mathbf{h}_t + \mathbf{b}_y)$$

### 2. LSTM Gating Architecture
The LSTM maintains a separate **Cell State $\mathbf{C}_t$** (long-term memory) and **Hidden State $\mathbf{h}_t$** (short-term working memory),
controlled by three continuous gating vectors bounded in $(0, 1)$ via Sigmoid activations:

```text
         ┌────────────────────────────────────────────────────────┐
  C_{t-1}───►( x )───────────────────( + )──────────────────────────► C_t
               ▲                       ▲
               │                       │
               │ Forget Gate           │ Input Candidate Gate
             ┌─┴─┐                   ┌─┴─┐
             │ f │                   │ i │ * C~
             └───┘                   └───┘
  h_{t-1}──┐   ▲                       ▲                    ┌───┐
           ├───┴───────────────────────┴────────────────────┤ o │──► h_t
  x_t ─────┘                                                └───┘
```

Equations at step $t$:
1. **Forget Gate $\mathbf{f}_t$**: Decides what fraction of long-term memory to discard:
   $$\mathbf{f}_t = \sigma(W_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)$$
2. **Input Gate $\mathbf{i}_t$**: Decides which new candidate values to store:
   $$\mathbf{i}_t = \sigma(W_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)$$
3. **Candidate Memory Cell $\tilde{\mathbf{C}}_t$**:
   $$\tilde{\mathbf{C}}_t = \tanh(W_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)$$
4. **Cell State Update $\mathbf{C}_t$** (Linear update prevents vanishing gradients):
   $$\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t$$
5. **Output Gate $\mathbf{o}_t$ and Hidden State $\mathbf{h}_t$**:
   $$\mathbf{o}_t = \sigma(W_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)$$
   $$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{C}_t)$$

---

## When to Use

- **Time-Series Forecasting**: Stock price dynamics, temperature forecasting, industrial telemetry sensor streams.
- **Audio & Speech Processing**: Acoustic modeling, voice recognition, phoneme classification.
- **Bidirectional Sequential Tasks**: Part-of-speech tagging and named entity recognition where future context informs current predictions.

---

## Example

Below is a complete PyTorch implementation of a Bidirectional LSTM for sequence classification:

```python
import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    """Multi-layer Bidirectional LSTM for sequence classification."""
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,          # Input shape: (Batch, Seq_Len, Embed_Dim)
            bidirectional=True,        # Processes sequence forward and backward
            dropout=0.3 if num_layers > 1 else 0.0
        )
        # Multiplied by 2 because bidirectional concatenates forward and backward states
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len)
        embeds = self.embedding(x)
        lstm_out, (hn, cn) = self.lstm(embeds)
        
        # Concatenate final forward and backward hidden states from top layer
        # hn shape: (num_layers * 2, batch, hidden_dim)
        forward_last = hn[-2, :, :]
        backward_last = hn[-1, :, :]
        final_context = torch.cat([forward_last, backward_last], dim=1)
        
        logits = self.fc(final_context)
        return logits

if __name__ == "__main__":
    model = BiLSTMClassifier(vocab_size=1000, embed_dim=64, hidden_dim=128, num_classes=3)
    dummy_seq = torch.randint(1, 1000, (8, 50)) # 8 samples, length 50
    out = model(dummy_seq)
    print("Input Batch Shape:", dummy_seq.shape)
    print("Output Logits Shape:", out.shape)
```

---

## Key Points & Pitfalls

### 1. Inability to Parallelize Across Time
Because state $\mathbf{h}_t$ depends strictly on state $\mathbf{h}_{t-1}$, training an RNN requires sequential execution step-by-step.
GPUs cannot parallelize across the sequence length dimension, causing massive training latency compared to Transformers.

### 2. Forget Gate Initialization Trick
Initialize the forget gate bias $\mathbf{b}_f$ to large positive values (e.g. $+1.0$ or $+2.0$).
This forces $\sigma(\mathbf{b}_f) \approx 1.0$ at the start of training, ensuring the network remembers historical context by default.

### 3. Bidirectional Leakage in Forecasting
Never use Bidirectional LSTMs for future forecasting tasks; accessing future tokens constitutes catastrophic temporal data leakage.

---