# Machine Learning Evaluation Metrics: Formulas & Guidelines

## 1. Classification Metrics

### Confusion Matrix Fundamentals
| Term | Full Name | Definition |
|---|---|---|
| **TP** | True Positive | Ground truth is 1, Model predicted 1 |
| **FP** | False Positive | Ground truth is 0, Model predicted 1 (Type I Error) |
| **FN** | False Negative | Ground truth is 1, Model predicted 0 (Type II Error) |
| **TN** | True Negative | Ground truth is 0, Model predicted 0 |

### Formulas
- **Accuracy**: $\frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$
- **Precision (Positive Predictive Value)**: $\frac{\text{TP}}{\text{TP} + \text{FP}}$
- **Recall (Sensitivity / True Positive Rate)**: $\frac{\text{TP}}{\text{TP} + \text{FN}}$
- **Specificity (True Negative Rate)**: $\frac{\text{TN}}{\text{TN} + \text{FP}}$
- **$F_1$-Score**: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}$
- **$F_\beta$-Score**: $(1 + \beta^2) \frac{\text{Precision} \times \text{Recall}}{(\beta^2 \times \text{Precision}) + \text{Recall}}$ ($\beta=2$ weights recall higher; $\beta=0.5$ weights precision higher)
- **Matthews Correlation Coefficient (MCC)**: $\frac{\text{TP}\cdot\text{TN} - \text{FP}\cdot\text{FN}}{\sqrt{(\text{TP}+\text{FP})(\text{TP}+\text{FN})(\text{TN}+\text{FP})(\text{TN}+\text{FN})}}$

---

## 2. Threshold-Independent Curve Metrics
- **ROC-AUC (Area Under Receiver Operating Characteristic)**:
  Integrates TPR vs FPR across all thresholds. Probability that a random positive sample ranks higher than a random negative sample.
- **PR-AUC (Area Under Precision-Recall Curve)**:
  Integrates Precision vs Recall. Preferred metric (reported alongside ROC-AUC) when the positive class is rare (e.g., $< 5\%$).

---

## 3. Regression Metrics
- **Mean Squared Error (MSE)**: $\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$
- **Root Mean Squared Error (RMSE)**: $\sqrt{\text{MSE}}$ (Interpretable in target units)
- **Mean Absolute Error (MAE)**: $\frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$ (Robust to outliers)
- **Mean Absolute Percentage Error (MAPE)**: $\frac{100\%}{N} \sum_{i=1}^N \left| \frac{y_i - \hat{y}_i}{y_i} \right|$
- **$R^2$ Score (Coefficient of Determination)**: $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$

---

## 4. Ranking & Information Retrieval Metrics
- **Precision@K**: Fraction of top-$K$ retrieved documents that are relevant.
- **Recall@K**: Fraction of all relevant documents captured in top-$K$ results.
- **Mean Reciprocal Rank (MRR)**: $\frac{1}{|Q|} \sum_{q=1}^{|Q|} \frac{1}{\text{rank}_q}$ (Measures where the *first* relevant result appears).
- **NDCG@K (Normalized Discounted Cumulative Gain)**: Accounts for graded relevance with logarithmic rank discount.

---

## 5. Clustering Metrics
- **Silhouette Coefficient**: $s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))} \in [-1, 1]$; higher is better (no ground-truth labels required).
- **Davies-Bouldin Index**: Average similarity between each cluster and its most similar cluster; lower is better (no ground-truth labels required).
- **Adjusted Rand Index (ARI)**: Agreement between predicted clusters and ground-truth labels, corrected for chance; $1$ is a perfect match and values near $0$ indicate random labeling.