# Matplotlib & Seaborn: Diagnostic Visualizations for ML

## Concept
Data visualization is not merely an aesthetic presentation tool; it is a critical diagnostic instrument
across all phases of the machine learning lifecycle. Visualizations expose subtle dataset corruptions,
label imbalances, multicollinearity, covariate shift, and systematic model residuals that scalar metrics
(such as aggregate accuracy or $R^2$) routinely hide.

Matplotlib is Python's foundational visualization engine, offering pixel-level control over figures,
axes, artists, and renderers. Seaborn builds on top of Matplotlib, providing a high-level statistical interface
tailored for tabular Pandas DataFrames with automatic confidence interval bootstrapping and structured multi-plot grids.

---

## Formula / Core Idea

### 1. The Object-Oriented Hierarchy
Matplotlib has two distinct execution interfaces:
1. **The Pyplot State Machine (`plt.plot`)**: Mimics MATLAB. Easy for quick interactive scripts but brittle for complex layouts.
2. **The Object-Oriented (OO) API (`fig, ax = plt.subplots`)**: Explicitly instantiates Figure and Axes objects.
   This is the recommended standard for production-grade, maintainable visualization code.

```text
+-----------------------------------------------------------------------+
| Figure Object (The Canvas / Window)                                   |
|  +---------------------------------+ +------------------------------+ |
|  | Axes 1 (Subplot 1)              | | Axes 2 (Subplot 2)           | |
|  |  - XAxis (Ticks, Labels, Scale) | |  - Line2D Artists            | |
|  |  - YAxis (Ticks, Labels, Scale) | |  - Legend Artist             | |
|  |  - Title, Grid, Spines          | |  - Colorbar Artist           | |
|  +---------------------------------+ +------------------------------+ |
+-----------------------------------------------------------------------+
```

### 2. Common Diagnostic Charts in Machine Learning

| Diagnostic Plot | Target Phenomenon | Actionable Insight |
|---|---|---|
| **Correlation Heatmap** | Feature Multicollinearity ($r > 0.85$) | Remove or combine redundant predictors |
| **Residuals vs Fitted** | Non-linearity & Heteroskedasticity | Apply log-transforms or use non-linear models |
| **Normal Q-Q Plot** | Residual Kurtosis / Heavy Tails | Identifies outliers or non-Gaussian error distributions |
| **ROC & PR Curves** | Decision Threshold Trade-offs | Choose optimal operating point under class imbalance |
| **Feature Importance** | Gini Impurity / Permutation Drop | Feature pruning and model interpretability |

---

## When to Use

- **Exploratory Data Analysis (EDA)**: Visualizing distributions (histograms, KDE), bivariate relationships (scatter plots),
  and category summaries (box/violin plots) before training models.
- **Hyperparameter Tuning Diagnostics**: Plotting validation curves (training vs validation loss across epochs or depths)
  to detect the exact onset of overfitting.
- **Model Error Auditing**: Plotting confusion matrices, calibration curves, and residual distributions to understand *where*
  the model makes erroneous predictions.

---

## Example

Below is a complete, publication-grade diagnostic script generating a multi-panel regression model audit figure:

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Configure professional aesthetics
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11

# 2. Generate synthetic ground truth, predictions, and residuals
rng = np.random.default_rng(42)
y_true = rng.uniform(10, 100, size=500)
noise = rng.normal(loc=0, scale=5.0 + 0.05 * y_true, size=500)
y_pred = y_true + noise
residuals = y_true - y_pred

# 3. Create 2x2 Subplot Grid using Object-Oriented API
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10), dpi=300)

# Subplot 1: Predictions vs Actuals
axes[0, 0].scatter(y_true, y_pred, alpha=0.6, edgecolors="k", s=30, color="#1f77b4")
ideal_line = np.linspace(10, 100, 100)
axes[0, 0].plot(ideal_line, ideal_line, "r--", linewidth=2, label="Ideal Identity (y = x)")
axes[0, 0].set_title("Actual vs. Predicted Values")
axes[0, 0].set_xlabel("Ground Truth Target ($y$)")
axes[0, 0].set_ylabel(r"Model Prediction ($\hat{y}$)")
axes[0, 0].legend()

# Subplot 2: Residuals vs Fitted (Heteroskedasticity Check)
axes[0, 1].scatter(y_pred, residuals, alpha=0.6, edgecolors="k", s=30, color="#ff7f0e")
axes[0, 1].axhline(0, color="black", linestyle="--", linewidth=1.5)
axes[0, 1].set_title("Residuals vs. Fitted Values")
axes[0, 1].set_xlabel(r"Fitted Values ($\hat{y}$)")
axes[0, 1].set_ylabel(r"Residuals ($y - \hat{y}$)")

# Subplot 3: Residual Distribution Histogram with KDE
sns.histplot(residuals, kde=True, ax=axes[1, 0], color="#2ca02c", edgecolor="black")
axes[1, 0].set_title("Residual Error Distribution")
axes[1, 0].set_xlabel("Residual Error")
axes[1, 0].set_ylabel("Density / Frequency")

# Subplot 4: Bivariate Correlation Heatmap
feature_matrix = np.column_stack([y_true, y_pred, residuals, noise])
corr = np.corrcoef(feature_matrix, rowvar=False)
labels = ["Target", "Pred", "Residual", "Noise"]
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1, 1],
            xticklabels=labels, yticklabels=labels, vmin=-1, vmax=1)
axes[1, 1].set_title("Metric Correlation Matrix")

# Clean layout adjustment
plt.tight_layout()
plt.savefig("diagnostic_report.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Diagnostic report saved successfully.")
```

---

## Key Points & Pitfalls

### 1. Memory Leaks in Interactive / Training Loops
Calling `plt.figure()` inside an epoch training loop without calling `plt.close(fig)` accumulates figure
objects in memory, eventually triggering memory exhaustion.
Always invoke `plt.close(fig)` after saving, or create figures with `matplotlib.figure.Figure` directly (pyplot does not track those).

### 2. High DPI and Bounding Box Truncation
When saving figures to disk, axis labels often get clipped off at image boundaries.
Pass `bbox_inches='tight'` to `plt.savefig()` to prevent clipping, and set `dpi=300` for crisp, print-quality output.

### 3. Choosing Color Palettes Wisely
- **Sequential Palettes** (e.g. `viridis`, `plasma`): Use for continuous metrics ranging from low to high.
- **Diverging Palettes** (e.g. `coolwarm`, `RdBu`): Use for zero-centered data (correlations, z-scores, residuals).
- **Qualitative Palettes** (e.g. `Set2`, `tab10`): Use for distinct categorical groups.

---