# Neural Networks

## Concept
Layers of connected neurons that learn complex non-linear patterns.

## Formula / Core Idea
Neuron: a = activation(w.x + b). Activations: ReLU, sigmoid, tanh, softmax.

## When to Use
Images, text, audio, and complex tabular problems with enough data.

## Example
```python
import torch.nn as nn
model = nn.Sequential(nn.Linear(10,32), nn.ReLU(), nn.Linear(32,1))
```

## Key Points
- Input, hidden and output layers
- Non-linear activations are essential
- Deeper networks learn more abstract features
- Needs more data and compute than classical ML

## My Notes
- (add your own notes here)
