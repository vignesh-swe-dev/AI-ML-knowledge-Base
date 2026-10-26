# Curated AI & Machine Learning Resources

## 📌 Overview
In the rapidly accelerating field of Artificial Intelligence, the primary obstacle to mastery is not a scarcity of information,
but an overwhelming abundance of low-signal, superficial tutorials. Navigating from beginner to senior machine learning
engineer requires prioritizing high-signal, first-principles textbooks, peer-reviewed foundational papers, rigorous university courses,
and battle-tested open-source repositories.

This curated compendium categorizes the gold-standard learning materials across textbooks, landmark research papers,
video lecture series, engineering blogs, and benchmark datasets.

---

## 📚 Premier Textbooks

### 1. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (3rd Edition)
- **Author**: Aurélien Géron
- **Focus**: A widely recommended applied textbook for machine learning engineers. Bridges high-level concepts directly to practical Python code.
- **Key Chapters**: Part I (End-to-End ML Project, Ensemble Learning, Dimensionality Reduction) and Part II (Neural Networks with Keras, Custom Models and Training with TensorFlow, Transformers, Autoencoders).

### 2. Deep Learning (Adaptive Computation and Machine Learning series)
- **Authors**: Ian Goodfellow, Yoshua Bengio, and Aaron Courville
- **Focus**: The definitive theoretical reference on deep learning, often referred to as the "Bible of Deep Learning".
- **Key Chapters**: Applied Math and Machine Learning Basics, Deep Feedforward Networks, Optimization for Deep Models, Convolutional Networks, Sequence Modeling.

### 3. Mathematics for Machine Learning
- **Authors**: Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong
- **Focus**: Rigorous linear algebra, analytic geometry, matrix decompositions, vector calculus, probability, and continuous optimization.
- **Availability**: Freely available online at [mml-book.com](https://mml-book.com).

### 4. Pattern Recognition and Machine Learning (PRML)
- **Author**: Christopher M. Bishop
- **Focus**: The classic Bayesian and probabilistic treatment of machine learning algorithms. Essential for probabilistic graphical models.

### 5. Reinforcement Learning: An Introduction (2nd Edition)
- **Authors**: Richard S. Sutton and Andrew G. Barto
- **Focus**: The definitive foundational text on Markov Decision Processes (MDPs), Dynamic Programming, Monte Carlo methods, Temporal-Difference learning, and Policy Gradients.

---

## 📄 Landmark Research Papers (Must-Read Canon)

| Year | Paper Title | Authors | Core Innovation |
|---|---|---|---|
| 2012 | **ImageNet Classification with Deep CNNs (AlexNet)** | Krizhevsky, Sutskever, Hinton | GPU acceleration, ReLU, Dropout; ignited modern Deep Learning |
| 2013 | **Distributed Representations of Words and Phrases and their Compositionality (Word2Vec)** | Mikolov et al. | Skip-Gram with Negative Sampling; dense continuous word vectors |
| 2014 | **Generative Adversarial Nets (GANs)** | Goodfellow et al. | Minimax two-player game between Generator and Discriminator |
| 2015 | **Deep Residual Learning for Image Recognition (ResNet)** | He, Zhang, Ren, Sun | Identity skip connections solving the degradation problem in very deep networks |
| 2017 | **Attention Is All You Need (Transformer)** | Vaswani et al. | Scaled dot-product self-attention replacing recurrence entirely |
| 2018 | **BERT: Pre-training of Deep Bidirectional Transformers** | Devlin et al. | Masked Language Modeling (MLM) and Next Sentence Prediction |
| 2020 | **Language Models are Few-Shot Learners (GPT-3)** | Brown et al. | In-context learning at scale; empirical validation of scaling laws |
| 2021 | **LoRA: Low-Rank Adaptation of Large Language Models** | Hu et al. | Low-rank decomposition matrices for parameter-efficient fine-tuning |
| 2022 | **FlashAttention: Fast and Memory-Efficient Exact Attention** | Dao et al. | Tiling and GPU SRAM kernel fusion reducing attention memory from $O(N^2)$ to $O(N)$ |
| 2023 | **Direct Preference Optimization (DPO)** | Rafailov et al. | Optimizes the policy directly on preference data without a separate reward model or reinforcement learning loop |

---

## 🎓 Premier Courses & Video Lectures

### 1. Stanford University Online Lectures
- **CS229**: *Machine Learning* (Andrew Ng) — Mathematical foundations of supervised learning, learning theory, and unsupervised learning.
- **CS231n**: *Deep Learning for Computer Vision* (Fei-Fei Li, Andrej Karpathy, Justin Johnson) — Landmark course covering backprop, CNNs, and vision architectures.
- **CS224n**: *Natural Language Processing with Deep Learning* (Christopher Manning) — Word vectors, RNNs, Transformers, and modern LLMs.

### 2. Practical & Intuitive Series
- **Andrej Karpathy's "Neural Networks: Zero to Hero"** (YouTube):
  Hands-on building of `micrograd` (autodiff from scratch), `makemore` (language modeling), and a full GPT decoder in pure PyTorch.
- **fast.ai: Practical Deep Learning for Coders** (Jeremy Howard):
  Top-down pragmatic approach teaching model fine-tuning, computer vision, tabular modeling, and diffusion models.
- **StatQuest with Josh Starmer** (YouTube):
  Clear, step-by-step visual explanations of statistics, logistic regression, trees, PCA, SVMs, and neural networks.
- **3Blue1Brown: Essence of Linear Algebra & Neural Networks** (YouTube):
  Geometric intuition for eigenvalues, matrix transformations, gradient descent, and backpropagation.

---

## 🌐 High-Signal Blogs, Newsletters & Community Hubs

- **Lil'Log (Lilian Weng)**: In-depth technical surveys on diffusion models, LLM agents, prompt engineering, and alignment.
- **Chip Huyen's Blog**: Production ML systems, real-time machine learning, feature stores, and LLM application design.
- **The Gradient**: Independent digital magazine featuring essays and interviews with AI researchers.
- **Hugging Face Papers**: Trending research papers linked to models, datasets, and code (the successor to Papers with Code, which Meta retired in July 2025).
- **Hugging Face Hub**: The central registry for open-source model weights, tokenizers, and datasets.

---

## 🛠️ Datasets & Competitive Practice Platforms

- **Kaggle**: Real-world competitive datasets, active discussion forums, and community notebooks.
- **OpenML**: Machine learning dataset repository with automated benchmarking APIs in Scikit-Learn.
- **Hugging Face Datasets**: Hundreds of thousands of NLP, audio, and vision datasets loadable with a single line of Python code.
- **Google Colab & Kaggle Notebooks**: Free access to cloud GPU/TPU compute environments for model experimentation.
