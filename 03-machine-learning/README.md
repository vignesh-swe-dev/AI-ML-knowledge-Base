# Module 03: Classical Machine Learning & Algorithms

## 📌 Module Overview
Classical machine learning encompasses the foundational supervised and unsupervised algorithms that power
the vast majority of enterprise production systems. While deep learning dominates unstructured data
(images, audio, natural language), tabular and structured business datasets remain the domain of
gradient-boosted trees, regularized linear models, and ensemble architectures.

This module provides a mathematically rigorous, code-backed deep dive into classical algorithms,
from statistical derivation to vectorized implementations and Scikit-Learn production pipelines.

---

## 📚 Syllabus & Topic Matrix

| File | Algorithm / Topic | Core Concepts | Industry Use Cases |
|---|---|---|---|
| [01-ml-overview.md](01-ml-overview.md) | ML Taxonomy & Paradigms | Inductive bias, ML lifecycle, No Free Lunch | Problem formulation, benchmark framing |
| [02-linear-regression.md](02-linear-regression.md) | Linear Regression & Regularization | OLS closed-form, Ridge, Lasso, ElasticNet | Econometrics, real estate valuation, forecasting |
| [03-logistic-regression.md](03-logistic-regression.md) | Logistic Regression & Softmax | Sigmoid, Log-Loss, Multi-class Softmax, Odds | Click-through rate (CTR), credit risk, churn |
| [04-decision-trees-random-forest.md](04-decision-trees-random-forest.md) | Trees & Random Forests | Gini, Entropy, Bagging, Random Subspaces | Tabular modeling, interpretability, fraud |
| [05-svm-knn.md](05-svm-knn.md) | SVM & K-Nearest Neighbors | Max-margin, RBF Kernel, KD-Trees, Distance | Anomaly detection, image embeddings, search |
| [06-clustering.md](06-clustering.md) | Unsupervised Clustering | K-Means++, DBSCAN, Silhouette Analysis | Customer segmentation, document clustering |
| [07-model-evaluation.md](07-model-evaluation.md) | Model Validation & Metrics | ROC-AUC, PR-AUC, F1, Stratified K-Fold | Benchmark auditing, threshold optimization |
| [08-feature-engineering.md](08-feature-engineering.md) | Feature Pipelines | Encoding, Imputation, Power Transforms | Production ETL, data leakage prevention |

---

## ⚖️ The Bias-Variance Tradeoff

For squared-error loss, the expected generalization error of a supervised model on unseen test data decomposes into three components:
$$\mathbb{E}[\text{Error}] = \text{Bias}^2 + \text{Variance} + \sigma^2$$

- **Bias**: Error stemming from overly simplistic assumptions in the learning algorithm (underfitting).
- **Variance**: Error stemming from excessive sensitivity to small fluctuations in the training dataset (overfitting).
- **Irreducible Error ($\sigma^2$)**: Inherent noise in the problem domain that cannot be eliminated by any model.

---

## 💡 Top 5 Technical Interview Questions (Classical ML)

### Q1: Why does $L_1$ regularization (Lasso) yield sparse weights while $L_2$ (Ridge) does not?
**Answer**: In constrained optimization (Lagrange multipliers), $L_1$ places a diamond-shaped constraint
($\sum |w_i| \le t$) on the parameter space, whose sharp corners align with coordinate axes where parameters equal zero.
The elliptical contours of the OLS loss function frequently intersect these corners, driving coefficients exactly to zero.
In contrast, $L_2$ imposes a smooth spherical constraint ($\sum w_i^2 \le t$), which shrinks coefficients asymptotically
toward zero but almost never sets them identically to zero.

### Q2: What is the Out-of-Bag (OOB) error in Random Forests, and why is it useful?
**Answer**: In Random Forests, each tree is trained on a bootstrap sample drawn with replacement from $N$ instances.
The probability of any sample *not* being chosen for a tree is $\lim_{N \to \infty} (1 - 1/N)^N = 1/e \approx 36.8\%$.
These unchosen samples are "Out-of-Bag" for that specific tree. Evaluating each instance using only the ensemble of trees
where it was OOB yields a nearly unbiased generalization estimate, comparable to cross-validation, without
the computational expense of running separate CV folds.

### Q3: How does the choice of distance metric impact K-Nearest Neighbors in high dimensions?
**Answer**: Due to the **Curse of Dimensionality**, the ratio of the distance between the nearest and farthest points
approaches 1 as dimensionality $D \to \infty$ under Euclidean distance. All points become equidistant, rendering proximity
meaningless. Mitigations include feature selection, dimensionality reduction (PCA), or using learned embeddings in which distances remain meaningful.

### Q4: When should you prioritize PR-AUC (Precision-Recall) over ROC-AUC?
**Answer**: Prioritize PR-AUC whenever evaluating models on **severely imbalanced datasets** (e.g., fraud detection with 0.1% positives).
ROC-AUC plots True Positive Rate against False Positive Rate. Because False Positive Rate has the huge True Negative count in its denominator
($\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$), millions of false alarms will barely shift the ROC curve. The PR curve
plots Precision vs Recall, directly penalizing False Positives regardless of how large True Negatives are.

### Q5: What is Data Leakage and how do you prevent it in Scikit-Learn?
**Answer**: Data leakage occurs when information from outside the training dataset (such as test set statistics) is
used to train the model, artificially inflating validation metrics. Examples include standardizing the entire dataset
before splitting into train/test, or imputing missing values using the global mean. Prevention: Always encapsulate
preprocessing and model estimators inside a `sklearn.pipeline.Pipeline`, ensuring `fit()` is called exclusively on training folds.

---

## 📝 Study Checklist & Milestones
- [ ] Derive the closed-form normal equation for Ordinary Least Squares regression.
- [ ] Implement Logistic Regression from scratch with vector cross-entropy and gradient descent.
- [ ] Build a Decision Tree classifier using Gini impurity recursion.
- [ ] Master Scikit-Learn Pipelines with `ColumnTransformer` for mixed-type tabular data.
- [ ] Implement K-Means and DBSCAN from scratch and evaluate clustering quality using Silhouette scores.

