# Production Machine Learning Project Template & Engineering Standard

## 1. Project Metadata & Overview
- **Project Name**: [e.g. End-to-End Customer Churn Prediction Engine]
- **Author / Lead**: [Your Name / Organization]
- **Target Completion Date**: [YYYY-MM-DD]
- **Problem Type**: [Binary Classification / Multi-Class / Regression / Object Detection / RAG]
- **Primary Business Objective**: [e.g. Reduce subscriber churn by 15% through proactive retention outreach.]
- **Business KPI**: [e.g. Net Churn Rate, Customer Lifetime Value (CLV), Monthly Recurring Revenue (MRR)]
- **Primary ML Metric**: [e.g. PR-AUC $\ge 0.82$, Precision at 90% Recall $\ge 0.75$]

---

## 2. Directory Architecture Standard

```text
project-root/
├── README.md               # Executive summary and quickstart guide
├── requirements.txt        # Pinned production dependencies
├── Dockerfile              # Containerization definition for API serving
├── Makefile                # Automation commands (make test, make lint, make train)
├── configs/                # YAML configuration files for hyperparameters
│   └── config.yaml
├── data/                   # Data directory (ignored in .gitignore)
│   ├── raw/                # Immutable original source files
│   ├── processed/          # Cleaned, standardized tabular data
│   └── interim/            # Intermediate pipeline caches
├── notebooks/              # Exploratory Data Analysis & experimentation
│   └── 01_eda_exploration.ipynb
├── src/                    # Source code package
│   ├── __init__.py
│   ├── data_loader.py      # Ingestion & schema validation
│   ├── features.py         # Feature engineering & transformers
│   ├── model.py            # Model training & hyperparameter search
│   ├── evaluate.py         # Validation metrics & diagnostic plots
│   └── app.py              # FastAPI inference endpoints
└── tests/                  # Unit tests and schema validation suites
    ├── test_data.py
    └── test_model.py
```

---

## 3. Data Ingestion & Quality Validation
- **Data Source**: [e.g. PostgreSQL Database / Snowflake Data Warehouse / S3 Bucket]
- **Observation Grain**: [e.g. One row per user per month]
- **Target Definition**: [e.g. 1 if user cancelled subscription within 30 days, else 0]
- **Missing Value Strategy**: [e.g. Numerical median imputation with missing indicator flags; Categorical 'Unknown']
- **Data Validation Rules**: [e.g. Age must be $\in [18, 100]$, Income must be $\ge 0$]

---

## 4. Feature Engineering & Preprocessing Pipeline
Document all transformations applied to raw features:
- **Categorical Variables**: One-Hot Encoding for low cardinality; Out-of-fold Target Encoding for high cardinality.
- **Numerical Scaling**: RobustScaler for skewed variables; StandardScaler for Gaussian features.
- **Domain Interaction Features**: Ratios, differences, polynomial terms, rolling moving averages.
- **Data Leakage Safeguards**: Transformers strictly fitted on training splits inside `sklearn.pipeline.Pipeline`.

---

## 5. Model Training & Validation Protocol
- **Validation Split Scheme**: [e.g. 5-Fold Stratified K-Fold / TimeSeriesSplit with 3-month horizon]
- **Baseline Model**: [e.g. Logistic Regression / DummyClassifier(strategy='most_frequent')]
- **Candidate Algorithms Evaluated**: [e.g. LightGBM, XGBoost, CatBoost, Random Forest]
- **Hyperparameter Tuning Strategy**: [e.g. Optuna Bayesian Optimization over 100 trials]
- **Selected Champion Model**: [e.g. LightGBM with depth=6, lr=0.03, subsample=0.8]

---

## 6. Evaluation Results & Performance Summary
*(The numbers below are illustrative placeholder values; replace them with your own results.)*

| Model Candidate | Train Accuracy | Val Accuracy | Val Precision | Val Recall | Val PR-AUC |
|---|---|---|---|---|---|
| **Dummy Baseline** | 0.850 | 0.850 | 0.000 | 0.000 | 0.150 |
| **Logistic Regression** | 0.862 | 0.860 | 0.620 | 0.450 | 0.580 |
| **Random Forest** | 0.940 | 0.895 | 0.760 | 0.680 | 0.770 |
| **Champion (LightGBM)** | **0.955** | **0.912** | **0.820** | **0.780** | **0.845** |

---

## 7. Model Serving & Deployment Specifications
- **Inference Paradigm**: [Real-time REST API / Batch Offline Scoring]
- **Target Latency SLA**: [e.g. $\le 25\text{ms}$ at p99]
- **Model Serialization**: `joblib.dump(pipeline, 'model_artifact.joblib')`
- **Monitoring & Drift**: Monitor input feature Kolmogorov-Smirnov test statistics and prediction output distributions.
