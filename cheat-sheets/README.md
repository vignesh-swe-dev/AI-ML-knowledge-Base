# Machine Learning & Data Science Cheat Sheets

## 📌 Overview
This directory contains high-density, production-tested quick reference cheat sheets designed for rapid syntax lookup,
algorithm selection during technical interviews, and day-to-day machine learning engineering.

---

## 📚 Cheat Sheet Directory

| File | Domain | Key Coverage |
|---|---|---|
| [numpy-pandas.md](numpy-pandas.md) | Scientific Computing & Tabular Data | Array slicing, broadcasting, vector math, DataFrame indexing, groupby, joins |
| [sklearn.md](sklearn.md) | Scikit-Learn API & Modeling | Estimators, transformers, ColumnTransformer, pipelines, grid search, metrics |
| [ml-metrics.md](ml-metrics.md) | Evaluation Metrics & Diagnostics | Classification, regression, clustering, ranking formulas & decision guides |
| [git-github.md](git-github.md) | Professional Version Control | Branching, interactive rebase, stashing, conflict resolution, Conventional Commits |

---

## ⚡ Quick Decision Heuristics

### Algorithm Selection Guide
- **Small Tabular ($N < 10,000$)**: Logistic / Ridge Regression, Random Forest, SVM with RBF Kernel.
- **Large Tabular ($N > 10,000$)**: LightGBM, XGBoost, CatBoost.
- **High-Dimensional Text ($D \gg N$)**: Linear SVM, Logistic Regression with $L_2$ penalty, Naive Bayes.
- **Unstructured Text / NLP**: Pretrained Transformers (RoBERTa for classification, Llama/GPT for generation).
- **Image / Spatial Data**: Convolutional Neural Networks (ResNet, ConvNeXt) or Vision Transformers (ViT).

### Metric Selection Guide
- **Balanced Classification**: Accuracy, ROC-AUC.
- **Imbalanced Classification**: PR-AUC, F1-Score, Cost-Weighted Misclassification Loss.
- **Regression with Normal Errors**: Mean Squared Error (MSE), RMSE, $R^2$.
- **Regression with Heavy Outliers**: Mean Absolute Error (MAE), Huber Loss.
