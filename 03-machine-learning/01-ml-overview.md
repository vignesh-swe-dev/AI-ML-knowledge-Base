# Machine Learning: Paradigms, Lifecycle & Inductive Bias

## Concept
Machine learning is the study of computer algorithms that improve automatically through experience.
Formally defined by Tom Mitchell (1997): *"A computer program is said to learn from experience E with respect
to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."*

Unlike rule-based expert systems where domain logic is explicitly coded by human programmers, machine learning systems
infer statistical mappings $f: \mathcal{X} \to \mathcal{Y}$ directly from empirical data. Understanding the core paradigms
(Supervised, Unsupervised, Semi-Supervised, Reinforcement Learning) and the operational lifecycle ensures that machine learning
projects solve genuine business needs rather than becoming fragile research prototypes.

---

## Formula / Core Idea

### 1. The Machine Learning Taxonomy
- **Supervised Learning**: Input pairs $(\mathbf{x}_i, y_i)$. Goal: learn mapping $f(\mathbf{x}) \approx y$. Tasks: Regression (continuous target), Classification (discrete categories).
- **Unsupervised Learning**: Unlabeled inputs $\mathbf{x}_i$. Goal: discover underlying structure, density, or low-dimensional manifolds. Tasks: Clustering, Dimensionality Reduction, Density Estimation.
- **Self-Supervised Learning**: Generates supervisory signals from unlabeled data itself. Tasks: Masked Language Modeling (BERT), Autoregressive next-token prediction (GPT), Contrastive Learning (SimCLR).
- **Reinforcement Learning**: Agent interacts with dynamic environment via states $s_t$, actions $a_t$, and scalar rewards $r_t$. Goal: maximize expected cumulative discounted return.

### 2. The Production Machine Learning Lifecycle (MLOps)
1. **Problem Formulation**: Framing the business objective into an ML task and defining business KPIs.
2. **Data Acquisition & Labeling**: Ingestion pipelines, schema validation, data quality checks.
3. **Exploratory Data Analysis & Feature Engineering**: Distribution auditing, missing value imputation, encoding, and scaling.
4. **Model Architecture Selection & Training**: Establishing simple heuristic baselines before scaling to complex ensembles.
5. **Offline Evaluation & Validation**: Rigorous cross-validation, slice-based evaluation, fairness auditing, and threshold tuning.
6. **Deployment, Monitoring & Retraining**: Containerized inference APIs, drift detection (data drift, concept drift).

### 3. Inductive Bias & No Free Lunch Theorem
- **Inductive Bias**: The set of prior assumptions a learning algorithm uses to predict outputs on unseen inputs.
- **No Free Lunch Theorem**: Averaged over all possible problems, no single learning algorithm outperforms any other on unseen data. In practice, an algorithm's assumptions must match the structure of the problem at hand.

---

## When to Use

- Use **Supervised Learning** when high-quality ground-truth labels exist and the goal is prediction.
- Use **Unsupervised Learning** when labels are absent or when discovering hidden customer groupings, anomalies, or topics.
- Use **Self-Supervised Pretraining** when massive unlabeled corpora exist to learn general representations before fine-tuning.

---

## Example

Below is a complete Python benchmarking harness comparing baseline models against a dummy heuristic:

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

X, y = make_classification(
    n_samples=5000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    weights=[0.85, 0.15],
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

models = {
    "Dummy Baseline (Most Frequent)": DummyClassifier(strategy="most_frequent"),
    "Linear Model (Logistic Regression)": LogisticRegression(max_iter=1000),
    "Tree Ensemble (Random Forest)": RandomForestClassifier(n_estimators=100, random_state=42),
}

for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    print(f"{name:<38} | Accuracy: {acc:.4f} | F1: {f1:.4f}")
```

---

## Key Points & Pitfalls
- Always establish a simple, interpretable baseline before exploring complex non-linear models.
- Distinguish between Data Drift (covariate shift) and Concept Drift (relationship change).
- Align loss functions and evaluation metrics strictly with business utility.

---

