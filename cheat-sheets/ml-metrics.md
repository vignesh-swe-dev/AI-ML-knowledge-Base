# ML Metrics Cheat Sheet

## Classification
- Accuracy = correct / total
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 * P * R / (P + R)
- ROC-AUC: ranking quality across thresholds

## Regression
- MAE = mean(|y - y_pred|)
- RMSE = sqrt(mean((y - y_pred)^2))
- R-squared: share of variance explained

## Which to use
- Imbalanced classes: precision, recall, F1, PR-AUC
- Costly false negatives: prioritise recall
- Costly false positives: prioritise precision
