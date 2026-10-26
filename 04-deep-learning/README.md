# Module 04: Deep Learning & Neural Architectures

## 📌 Module Overview
Deep Learning has driven the modern renaissance in artificial intelligence, delivering state-of-the-art results
across computer vision, natural language understanding, speech recognition, and generative modeling.
Unlike classical machine learning models that rely heavily on manual feature engineering, deep neural networks
learn hierarchical feature representations directly from raw, unstructured data (pixels, audio waveforms, character sequences).

This module provides a comprehensive, ground-up architectural study of deep learning:
from computational graphs and automatic differentiation to convolutional neural networks, sequence modeling,
and the foundational Transformer self-attention architecture that powers contemporary foundation models.

---

## 📚 Syllabus & Topic Matrix

| File | Subject | Core Architecture | Primary Domains |
|---|---|---|---|
| [01-neural-networks.md](01-neural-networks.md) | Multi-Layer Perceptrons & Activations | Dense Feed-Forward Networks, ReLU, GELU, He Init | Universal function approximation, tabular deep learning |
| [02-backpropagation-optimizers.md](02-backpropagation-optimizers.md) | Autodiff & Optimizers | Reverse-mode Autodiff, Computational Graphs, AdamW | Loss minimization, convergence stability, training dynamics |
| [03-cnn.md](03-cnn.md) | Convolutional Neural Networks | Convolutions, Pooling, ResNet Skip Connections | Computer vision, spatial processing, image classification |
| [04-rnn-lstm.md](04-rnn-lstm.md) | Recurrent & Sequential Networks | Hidden States, BPTT, Gated Memory Cells (LSTM, GRU) | Time-series forecasting, audio signals, sequential text |
| [05-transformers.md](05-transformers.md) | The Transformer Architecture | Scaled Dot-Product Attention, Multi-Head Attention, RoPE | Foundation models, LLMs, Vision Transformers (ViT) |

---

## 🏛️ Evolution of Neural Architectures
- 1958: Perceptron (Rosenblatt) — Single linear decision boundary.
- 1986: Multi-Layer Perceptron (Rumelhart et al.) — Backpropagation enables training non-linear hidden layers.
- 1998: LeNet-5 (LeCun et al.) — Convolutional kernels exploit spatial translation equivariance.
- 2012: AlexNet (Krizhevsky et al.) — GPU acceleration + ReLU ignites modern Deep Learning.
- 2015: ResNet (He et al.) — Residual skip connections enable 100+ layer deep networks.
- 2017: Transformer (Vaswani et al.) — Self-attention replaces recurrence; enables parallel sequence modeling.
- 2020+: Foundation Models (GPT, Llama, ViT) — Massive self-supervised pretraining at global compute scale.

---

## 💡 Top Technical Interview Questions (Deep Learning)
### Q1: What is the Vanishing Gradient problem and how do modern architectures solve it?
**Answer**: In deep networks, the chain rule multiplies derivatives across layers. Saturating activations (Sigmoid, Tanh)
have derivatives $\le 0.25$, causing gradients to vanish exponentially. Solved by ReLU/GELU, Residual Skip Connections, and Layer/Batch Normalization.

### Q2: Why is AdamW preferred over standard Adam with L2 regularization?
**Answer**: In Adam, $L_2$ regularization is divided by historical gradient scale $\sqrt{\mathbf{v}_t}$, distorting regularization.
AdamW decouples weight decay, applying it directly to parameters.

---

## 📝 Study Checklist & Milestones
- [ ] Implement forward and backward passes of an MLP in pure NumPy.
- [ ] Build and train a Convolutional Neural Network in PyTorch with DataLoaders.
- [ ] Implement an LSTM cell from scratch using basic tensor operations.
- [ ] Code Multi-Head Self-Attention in PyTorch and verify output tensor shapes.
- [ ] Fine-tune a pretrained Vision Transformer or BERT model using Hugging Face.
