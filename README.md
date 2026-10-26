# 🧠 AI & Machine Learning Study Hub

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Actively_Maintained-success?style=flat)](#-learning-roadmap--progress-tracker)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Format](https://img.shields.io/badge/Notes-Markdown-lightgrey?logo=markdown&logoColor=white)](#-repository-structure)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#-contributing)

<p align="center">
  <b>A comprehensive, first-principles roadmap and reference repository covering Python, Applied Mathematics, Classical Machine Learning, Deep Learning, and Modern NLP / LLMs.</b>
</p>

[Curriculum](#-curriculum--knowledge-base) •
[Repository Structure](#-repository-structure) •
[Note Format](#-standard-note-framework) •
[Roadmap Tracker](#-learning-roadmap--progress-tracker) •
[Cheat Sheets](#-cheat-sheets--quick-references) •
[Hands-On Projects](#-hands-on-projects) •
[Getting Started](#-getting-started) •
[FAQ](#-frequently-asked-questions)

---

</div>

## 📌 Overview & Engineering Philosophy

This repository is a structured, production-focused compendium of notes, mathematical foundations, and code implementations
across the Artificial Intelligence and Machine Learning landscape.

Instead of fragmented tutorials or superficial overviews, every topic in this repository is documented with mathematical rigor,
conceptual intuition, practical decision criteria, and self-contained, reproducible Python code snippets.

### 🎯 Key Architectural Principles
- **First-Principles Derivations**: Understand *why* an algorithm works mathematically before jumping into high-level APIs.
- **Predictable 5-Point Schema**: Every single note adheres to an identical, clean template for effortless navigation.
- **Theory to Production**: Complete journey from raw mathematical equations to NumPy implementations and PyTorch production modules.
- **Zero Fluff**: Quick-lookup cheat sheets, metric selection guides, and real-world project blueprints.

---

## 📚 Curriculum & Knowledge Base

| # | Module | Core Concepts & Topics | Direct Note Links |
|---|---|---|---|
| **01** | [**Python Basics & Scientific Stack**](01-python-basics/) | Core syntax, vectorization, tabular manipulation, data visualization | • [Python Fundamentals](01-python-basics/01-python-fundamentals.md)<br>• [NumPy](01-python-basics/02-numpy.md)<br>• [Pandas](01-python-basics/03-pandas.md)<br>• [Matplotlib](01-python-basics/04-matplotlib.md) |
| **02** | [**Math for Machine Learning**](02-math-for-ml/) | Vector spaces, matrices, probabilistic inference, calculus & gradient descent | • [Linear Algebra](02-math-for-ml/01-linear-algebra.md)<br>• [Probability & Statistics](02-math-for-ml/02-probability-statistics.md)<br>• [Calculus & Optimization](02-math-for-ml/03-calculus-optimization.md) |
| **03** | [**Classical Machine Learning**](03-machine-learning/) | Supervised & unsupervised learning, ensemble methods, clustering, validation | • [ML Overview](03-machine-learning/01-ml-overview.md)<br>• [Linear Regression](03-machine-learning/02-linear-regression.md)<br>• [Logistic Regression](03-machine-learning/03-logistic-regression.md)<br>• [Decision Trees & Random Forest](03-machine-learning/04-decision-trees-random-forest.md)<br>• [SVM & KNN](03-machine-learning/05-svm-knn.md)<br>• [Clustering (K-Means & DBSCAN)](03-machine-learning/06-clustering.md)<br>• [Model Evaluation & Metrics](03-machine-learning/07-model-evaluation.md)<br>• [Feature Engineering](03-machine-learning/08-feature-engineering.md) |
| **04** | [**Deep Learning**](04-deep-learning/) | Perceptrons, backpropagation, CNNs, sequential models, attention mechanisms | • [Neural Networks](04-deep-learning/01-neural-networks.md)<br>• [Backpropagation & Optimizers](04-deep-learning/02-backpropagation-optimizers.md)<br>• [Convolutional Neural Networks (CNN)](04-deep-learning/03-cnn.md)<br>• [RNNs & LSTMs](04-deep-learning/04-rnn-lstm.md)<br>• [Transformers & Attention](04-deep-learning/05-transformers.md) |
| **05** | [**Natural Language Processing**](05-nlp/) | Text tokenization, vector embeddings, fine-tuning, large language models | • [Text Preprocessing](05-nlp/01-text-preprocessing.md)<br>• [Word & Sentence Embeddings](05-nlp/02-embeddings.md)<br>• [LLM Fundamentals & Prompting](05-nlp/03-llms-basics.md) |
| **06** | [**Hands-on Projects**](06-projects/) | Real-world problem statements, pipeline blueprints, and implementation templates | • [Standardized Project Template](06-projects/project-template.md)<br>• [Project Catalog & Ideas](06-projects/README.md) |
| **—** | [**Cheat Sheets**](cheat-sheets/) | Rapid-reference cheat sheets for interviews and day-to-day development | • [NumPy & Pandas](cheat-sheets/numpy-pandas.md)<br>• [Scikit-Learn](cheat-sheets/sklearn.md)<br>• [ML Metrics](cheat-sheets/ml-metrics.md)<br>• [Git & GitHub](cheat-sheets/git-github.md) |
| **—** | [**Curated Resources**](resources/) | High-signal textbooks, accredited courses, benchmark papers, and YouTube creators | • [Recommended Reading & Courses](resources/README.md) |

---

## 📂 Repository Structure

```text
ai-ml-notes/
├── 01-python-basics/       # Core Python syntax, NumPy vectorization, Pandas, Matplotlib
├── 02-math-for-ml/         # Linear Algebra, Probability & Statistics, Multivariable Calculus
├── 03-machine-learning/    # Supervised, Unsupervised, Tree Ensembles, Evaluation & Metrics
├── 04-deep-learning/       # Neural Networks, Optimizers, CNNs, RNNs/LSTMs, Transformers
├── 05-nlp/                 # Preprocessing, Word & Sentence Embeddings, LLMs
├── 06-projects/            # Hands-on Project Ideas and Standardized Project Template
├── cheat-sheets/           # Rapid syntax lookups (NumPy/Pandas, Sklearn, Metrics, Git)
├── resources/              # Curated books, courses, papers, and video tutorials
└── LICENSE                 # Open-source MIT License
```

---

## 📐 Standard Note Framework

To maintain predictability and quality across the entire study hub, every single note adheres strictly to a **5-Point Schema**:

```markdown
# [Topic Title]

## Concept
Intuitive, plain-English explanation of what problem this technique solves.

## Formula / Core Idea
The underlying mathematics, architecture diagrams, loss functions, or algorithms.

## When to Use
Trade-offs, assumptions, real-world suitability, and failure modes.

## Example
Minimal, reproducible, self-contained Python / library snippet.

## Key Points & Pitfalls
- Production tips, common interview questions, and subtle edge cases.
```

---

## 🗺️ Learning Roadmap & Progress Tracker

### Phase 1: Foundations & Scientific Computing
- [x] Python core syntax, data structures, and comprehensions
- [x] NumPy multidimensional arrays, vectorization, and broadcasting
- [x] Pandas DataFrame manipulation, grouping, and feature cleaning
- [x] Matplotlib & Seaborn exploratory data visualization

### Phase 2: Mathematics & Statistics for AI
- [x] Linear Algebra (vectors, matrices, dot products, eigenvalues/eigenvectors)
- [x] Probability & Distributions (Gaussian, Bayes' theorem, expectation, variance)
- [x] Calculus & Optimization (partial derivatives, gradients, SGD, Adam)

### Phase 3: Classical Machine Learning
- [x] Linear Regression (Ordinary Least Squares, Ridge, Lasso)
- [x] Logistic Regression & Multi-Class Softmax Classification
- [x] Decision Trees, Bagging, and Random Forests
- [x] Support Vector Machines (SVM) & K-Nearest Neighbors (KNN)
- [x] Unsupervised Clustering (K-Means, DBSCAN, Hierarchical)
- [x] Model Evaluation (Cross-Validation, ROC-AUC, Precision-Recall, F1)
- [x] Feature Engineering & Leak-Proof Preprocessing pipelines

### Phase 4: Deep Learning & Neural Architectures
- [x] Multi-Layer Perceptrons (MLPs) & Activation Functions (ReLU, GELU, Swish)
- [x] Computational Graphs, Reverse-Mode Autodiff & AdamW Optimization
- [x] Convolutional Neural Networks (CNN) & ResNet Skip Connections
- [x] Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM)
- [x] Transformer Architecture (Self-Attention, Multi-Head Attention, RoPE)

### Phase 5: NLP & Generative AI
- [x] Text preprocessing & Byte-Pair Encoding (BPE) subword tokenization
- [x] Word & Sentence Embeddings (TF-IDF, Word2Vec, SBERT)
- [x] Large Language Models (LLMs), Decoding strategies, LoRA fine-tuning

### Phase 6: Hands-on Projects & Deployment
- [x] Advanced Tabular House Price Regression
- [x] Real-Time Spam & Phishing Detection API
- [x] Customer Segmentation & Lifetime Value Engine
- [x] Handwritten Digit & Mathematical Formula Recognition
- [x] Aspect-Based E-Commerce Sentiment Analysis
- [x] Enterprise Knowledge Retrieval-Augmented Generation (RAG) System

---

## 🛠️ Hands-On Projects

The [`06-projects/`](06-projects/) module features complete roadmaps for end-to-end applications:
1. **House Price Prediction**: Regression, feature engineering, and regularization.
2. **Spam Email Classifier**: NLP pipeline, TF-IDF vectorization, and FastAPI serving.
3. **Customer Segmentation**: Unsupervised clustering (K-Means, PCA dimensionality reduction).
4. **Handwritten Digit Recognition**: Computer Vision using Convolutional Neural Networks (PyTorch).
5. **Customer Sentiment Analysis**: Sequential models / Transformer embeddings on real reviews.
6. **RAG Knowledge Assistant**: Document ingestion, vector store retrieval, and LLM synthesis.

> [!TIP]
> When starting any project, copy [`06-projects/project-template.md`](06-projects/project-template.md) into your project folder to maintain clean documentation standards.

---

## ⚡ Cheat Sheets & Quick References

- 📊 **[NumPy & Pandas](cheat-sheets/numpy-pandas.md)**: Array indexing, DataFrame slicing, aggregations, and joins.
- 🤖 **[Scikit-Learn](cheat-sheets/sklearn.md)**: Estimator API, Pipeline construction, hyperparameter search, and preprocessing.
- 🎯 **[ML Metrics](cheat-sheets/ml-metrics.md)**: Confusion matrix, Precision vs Recall, ROC-AUC, RMSE, MAE.
- 🐙 **[Git & GitHub](cheat-sheets/git-github.md)**: Branching models, rebasing, stash management, and commit conventions.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/vignesh-swe-dev/AI-ML-knowledge-Base.git
cd AI-ML-knowledge-Base
```

### 2. Set Up a Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Activate on macOS / Linux:
source .venv/bin/activate
```

### 3. Recommended Packages
```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch torchvision transformers jupyterlab
```

### 4. Recommended Markdown Viewers
- **VS Code**: Install *Markdown All in One* and *Markdown Preview Enhanced* extensions.
- **Obsidian**: Open the repository folder directly as an Obsidian Vault for interactive graph views and bidirectional linking.
- **GitHub**: Native web rendering with table-of-contents support.

---

## ❓ Frequently Asked Questions (FAQ)

### Q: In what order should I study these notes?
**Answer**: Follow the numerical order: `01-python-basics` $\to$ `02-math-for-ml` $\to$ `03-machine-learning` $\to$ `04-deep-learning` $\to$ `05-nlp` $\to$ `06-projects`. This builds prerequisite knowledge systematically.

### Q: Are the code snippets runnable without modification?
**Answer**: Yes. Every code block is written as a self-contained, reproducible Python snippet containing synthetic data generation or standard benchmark datasets.

### Q: Can I contribute additional topics or fixes?
**Answer**: Absolutely! Contributions, typo corrections, and notes expansions are warmly welcomed. See the [Contributing Guidelines](#-contributing) below.

---

## 🤝 Contributing

Contributions, topic suggestions, and corrections are welcome!
1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/new-topic`).
3. Follow the [Standard Note Framework](#-standard-note-framework).
4. Commit your changes (`git commit -m "docs: add note on Gradient Boosting"`).
5. Push to the branch (`git push origin feature/new-topic`) and open a Pull Request.

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE) — feel free to use these notes for personal study, interview preparation, and reference.
