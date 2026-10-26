# Convolutional Neural Networks (CNNs) & Computer Vision

## Concept
Convolutional Neural Networks (CNNs) are specialized deep learning architectures designed for processing data
with a known, grid-like spatial topology, such as 2D digital images, 2D audio spectrograms, or 1D signals.

Standard fully connected MLPs fail catastrophically on high-resolution image data for two reasons:
1. **Parameter Explosion**: Connecting an MLP directly to a $1024 \times 1024 \times 3$ image produces over 3 million weights per neuron.
2. **Spatial Invariance Ignorance**: An MLP treats neighboring pixels identically to distant pixels, ignoring spatial geometry.

CNNs solve this by incorporating two powerful inductive biases:
- **Local Receptive Fields**: Neurons compute activations only over local spatial windows.
- **Shared Kernel Weights**: The same convolutional filter is swept across the entire image, providing **translation equivariance**:
  if an object shifts in the image, its feature map representation shifts by the exact same spatial translation.

---

## Formula / Core Idea

### 1. The 2D Discrete Convolution Operation
For an input image $I \in \mathbb{R}^{H \times W}$ and a kernel filter $K \in \mathbb{R}^{k_h \times k_w}$:
$$S(i, j) = \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} I(i + m, j + n) K(m, n)$$
(Strictly, this is cross-correlation, which is what deep learning libraries implement under the name "convolution".)

Output Spatial Dimension Formula:
$$H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} - K_h + 2P}{S} \right\rfloor + 1$$
Where $P$ is the padding size and $S$ is the stride step.

### 2. Residual Connections (ResNet)
As neural networks grew beyond 20 layers, accuracy degraded due to the optimization difficulty of identity mappings.
He et al. (2015) introduced the **Residual Block**:
$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$

```text
             Input x
                │
        ┌───────┴───────┐
        │               │ (Identity Shortcut)
      [Conv]            │
      [BatchNorm]       │
      [ReLU]            │
      [Conv]            │
      [BatchNorm]       │
        │               │
        └───────►( + )◄─┘
                  │
                [ReLU]
                  │
               Output y
```

During backpropagation, the gradient decomposes as:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \left( \frac{\partial \mathcal{F}}{\partial \mathbf{x}} + I \right)$$
The identity term $+ I$ acts as a gradient superhighway, allowing gradients to propagate directly to early layers unattenuated.

---

## When to Use

- **Computer Vision**: Image classification, semantic segmentation, object detection (YOLO), medical imaging.
- **1D Signal Processing**: Audio spectrogram classification, seismic signal processing.
- **Feature Extractors**: Backbone encoders for multi-modal Vision-Language Models (CLIP).

---

## Example

Below is a complete PyTorch implementation of a modern Residual Convolutional Block:

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """Standard ResNet residual block with identity skip shortcut."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual identity addition
        return self.relu(x + self.conv_block(x))

class MiniResNet(nn.Module):
    """Compact Residual CNN for image classification."""
    
    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResidualBlock(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.res2 = ResidualBlock(64)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = self.layer1(out)
        out = self.pool(out)
        out = self.layer2(out)
        out = self.res2(out)
        out = self.gap(out)
        out = torch.flatten(out, 1)
        return self.classifier(out)

if __name__ == "__main__":
    model = MiniResNet(in_channels=3, num_classes=10)
    dummy_input = torch.randn(4, 3, 32, 32)
    logits = model(dummy_input)
    print("Input Tensor Shape: ", dummy_input.shape)
    print("Output Logits Shape:", logits.shape)
```

---

## Key Points & Pitfalls

### 1. Convolution vs Cross-Correlation
Technically, deep learning libraries implement cross-correlation (no kernel flipping). Since kernels are learned
via backpropagation, kernel orientation is completely irrelevant.

### 2. Global Average Pooling (GAP) Over Flattening
Traditional networks flattened feature maps into massive linear layers (e.g. VGG had 100M+ classifier weights).
Modern CNNs use Global Average Pooling (`AdaptiveAvgPool2d((1, 1))`), reducing spatial maps to a single vector and
dramatically reducing parameter counts (VGG-16's fully connected layers hold roughly 120M of its 138M parameters).

### 3. Receptive Field Expansion
Stacking two $3 \times 3$ convolutions has an effective receptive field of $5 \times 5$, but requires fewer parameters
and includes an intermediate non-linearity.

---

