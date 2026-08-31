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
[Getting Started](#-getting-started)

---

</div>

## 📌 Overview

This repository is a structured, production-focused compendium of notes, mathematical foundations, and code implementations across the Artificial Intelligence and Machine Learning landscape. 

Instead of fragmented tutorials or superficial overviews, every topic is documented with clarity, rigorous mathematical intuition, practical decision criteria, and self-contained code snippets.

### 🎯 Key Highlights

- **First-Principles Approach**: Understand *why* an algorithm works before jumping into implementation.
- **Consistent 5-Point Architecture**: Predictable, clean note template across all topics.
- **Theory to Practice**: From derivation to Scikit-Learn, PyTorch, and Hugging Face implementations.
- **Zero Fluff**: Quick-lookup cheat sheets, metric selection guides, and real-world project outlines.

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

To maintain consistency and readability across the entire repository, every note adheres to a battle-tested **5-Point Schema**:

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

## My Notes
- Personal experiments, observations, and ongoing takeaways.
```

---

## 🗺️ Learning Roadmap & Progress Tracker

Use this checklist to track your self-study or course progress:

### Phase 1: Foundations & Scientific Computing
- [ ] Python core syntax, data structures, and comprehensions
- [ ] NumPy multidimensional arrays, vectorization, and broadcasting
- [ ] Pandas DataFrame manipulation, grouping, and feature cleaning
- [ ] Matplotlib & Seaborn exploratory data visualization

### Phase 2: Mathematics & Statistics for AI
- [ ] Linear Algebra (vectors, matrices, dot products, eigenvalues/eigenvectors)
- [ ] Probability & Distributions (Gaussian, Bayes' theorem, expectation, variance)
- [ ] Calculus & Optimization (partial derivatives, gradients, SGD, Adam)

### Phase 3: Classical Machine Learning
- [ ] Linear Regression (Ordinary Least Squares, Ridge, Lasso)
- [ ] Logistic Regression & Decision Boundaries
- [ ] Decision Trees, Bagging, and Random Forests
- [ ] Boosting algorithms (Gradient Boosting, XGBoost, LightGBM)
- [ ] Support Vector Machines (SVM) & K-Nearest Neighbors (KNN)
- [ ] Unsupervised Clustering (K-Means, Hierarchical, DBSCAN)
- [ ] Feature Engineering & Preprocessing pipelines
- [ ] Model Evaluation (Cross-Validation, ROC-AUC, Precision-Recall, F1)

### Phase 4: Deep Learning & Neural Architectures
- [ ] Multi-Layer Perceptrons (MLPs) & Activation Functions (ReLU, GELU, Sigmoid)
- [ ] Loss functions (Cross-Entropy, MSE) & Backpropagation mechanics
- [ ] Optimization techniques (Momentum, RMSprop, AdamW, Learning Rate Schedulers)
- [ ] Convolutional Neural Networks (CNN) for Computer Vision
- [ ] Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM)
- [ ] Transformer Architecture (Self-Attention, Multi-Head Attention, Positional Encoding)

### Phase 5: NLP & Generative AI
- [ ] NLP preprocessing (Tokenization, Lemmatization, Stopwords)
- [ ] Vector representations (TF-IDF, Word2Vec, Dense Embeddings)
- [ ] Large Language Models (LLMs), Decoding strategies, In-Context Learning
- [ ] Retrieval-Augmented Generation (RAG) foundations

### Phase 6: End-to-End Implementation
- [ ] Build end-to-end regression baseline
- [ ] Build end-to-end classification baseline
- [ ] Deploy first ML/NLP inference API or interactive dashboard

---

## 🛠️ Hands-On Projects

The `06-projects/` module features structured projects ranging from foundational to production-grade:

1. **House Price Prediction**: Regression, feature engineering, and regularization.
2. **Spam Email Classifier**: NLP pipeline, TF-IDF vectorization, and naive Bayes / logistic regression.
3. **Customer Segmentation**: Unsupervised clustering (K-Means, PCA dimensionality reduction).
4. **Handwritten Digit Recognition**: Computer Vision using Convolutional Neural Networks (PyTorch).
5. **Customer Sentiment Analysis**: Sequential models / Transformer embeddings on real reviews.
6. **RAG Knowledge Assistant**: Document ingestion, vector store retrieval, and LLM synthesis.

> [!TIP]
> When starting any project, copy [`06-projects/project-template.md`](06-projects/project-template.md) into your project folder to maintain clean documentation standards.

---

## ⚡ Cheat Sheets & Quick References

When you need immediate syntax during coding or interview prep:

- 📊 **[NumPy & Pandas](cheat-sheets/numpy-pandas.md)**: Array indexing, DataFrame slicing, aggregations, and joins.
- 🤖 **[Scikit-Learn](cheat-sheets/sklearn.md)**: Estimator API, Pipeline construction, grid search, and preprocessing.
- 🎯 **[ML Metrics](cheat-sheets/ml-metrics.md)**: Confusion matrix, Precision vs Recall, ROC-AUC, RMSE, MAE.
- 🐙 **[Git & GitHub](cheat-sheets/git-github.md)**: Branching models, rebasing, stash management, and commit conventions.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/ai-ml-notes.git
cd ai-ml-notes
```

### 2. Set Up a Virtual Environment (Optional, for running examples)
```bash
# Using venv
python -m venv .venv

# Activate on Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Activate on macOS / Linux:
source .venv/bin/activate
```

### 3. Recommended Packages
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyterlab
```

### 4. Recommended Markdown Viewers
- **VS Code**: Install *Markdown All in One* and *Markdown Preview Enhanced* extensions.
- **Obsidian**: Open the repository folder directly as an Obsidian Vault for interactive graph views and bidirectional linking.
- **GitHub**: Native web rendering with table-of-contents support.

---

## 📖 Curated Resources & Credits

This knowledge repository draws insights from industry-standard materials:

- **Courses**:
  - *Machine Learning & Deep Learning Specializations* by Andrew Ng (Coursera / DeepLearning.AI)
  - *Practical Deep Learning for Coders* by fast.ai
  - *Neural Networks: Zero to Hero* by Andrej Karpathy
  - *Hugging Face NLP Course*
- **Textbooks**:
  - *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (Aurélien Géron)
  - *Deep Learning* (Ian Goodfellow, Yoshua Bengio, Aaron Courville)
  - *Mathematics for Machine Learning* (Marc Peter Deisenroth et al.)
- **Visual & Intuitive Explanations**:
  - *StatQuest with Josh Starmer*
  - *3Blue1Brown (Essence of Linear Algebra & Neural Networks)*

For full links and notes, see [resources/README.md](resources/README.md).

---

## 🤝 Contributing

Contributions, topic suggestions, and corrections are welcome!
1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/new-topic`).
3. Follow the [Standard Note Framework](#-standard-note-framework).
4. Commit your changes (`git commit -m "Add note on Gradient Boosting"`).
5. Push to the branch (`git push origin feature/new-topic`) and open a Pull Request.

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE) — feel free to use these notes for personal study, interview preparation, and reference.
