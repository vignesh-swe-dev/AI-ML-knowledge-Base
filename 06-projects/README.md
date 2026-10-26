# Module 06: Hands-On Machine Learning Projects

## 📌 Module Overview
Theory without practice is empty; practice without theory is blind. While theoretical derivations and toy code
snippets establish conceptual clarity, true engineering competency requires building end-to-end systems that ingest
messy data, validate schemas, engineer robust features, optimize models, and expose predictions via production APIs.

This module outlines six comprehensive, industry-grade projects spanning regression, classification, clustering,
computer vision, natural language processing, and modern foundation model systems.
Each project is designed to be completed using the standardized [project-template.md](project-template.md).

---

## 📚 Project Catalog & Architecture Blueprints

### Project 1: Advanced Tabular House Price Regression
- **Domain**: Real Estate Valuation & Predictive Econometrics
- **Core Techniques**: Log-transforms on skewed targets, Target Encoding with cross-validation smoothing, Ridge/Lasso regularization, LightGBM/XGBoost gradient boosting, and Stacking Ensembles.
- **Evaluation Metrics**: Root Mean Squared Logarithmic Error (RMSLE), Mean Absolute Percentage Error (MAPE).
- **Deliverables**: Trained ensemble pipeline with SHAP interpretability explanations and feature importance plots.

### Project 2: Real-Time Spam & Phishing Detection API
- **Domain**: Cybersecurity & Text Classification
- **Core Techniques**: Subword BPE tokenization, TF-IDF vs fine-tuned DistilBERT embeddings, class imbalance weighting, PR-AUC threshold optimization.
- **Evaluation Metrics**: Precision-Recall AUC, Precision at 99% Recall (minimizing false alarms).
- **Deliverables**: FastAPI microservice containerized with Docker, responding in under 15ms per request.

### Project 3: Customer Lifetime Value (CLV) & Behavioral Segmentation
- **Domain**: E-Commerce & Growth Marketing
- **Core Techniques**: Recency-Frequency-Monetary (RFM) feature engineering, Pareto/NBD statistical models, K-Means++ clustering with Silhouette analysis, UMAP 2D manifold projection.
- **Evaluation Metrics**: Silhouette Coefficient, Davies-Bouldin Index, Within-Cluster Variance.
- **Deliverables**: Interactive customer segment dashboard with actionable persona summaries.

### Project 4: Handwritten Digit & Mathematical Formula Recognition
- **Domain**: Computer Vision & Document Digitization
- **Core Techniques**: Convolutional Neural Networks (CNN) with Residual shortcuts in PyTorch, spatial data augmentation (random affine, elastic distortions), Batch Normalization, ONNX model export.
- **Evaluation Metrics**: Top-1 Accuracy, Macro F1-Score, Inference Latency on CPU/GPU.
- **Deliverables**: Web-based interactive canvas (Gradio or Streamlit) where users draw digits and receive real-time classification probabilities.

### Project 5: Aspect-Based E-Commerce Sentiment Analysis
- **Domain**: Customer Experience & Voice-of-Customer Intelligence
- **Core Techniques**: DeBERTa-v3 token classification, multi-task learning (predicting sentiment polarity and aspect category simultaneously), Focal Loss for severe class imbalance.
- **Evaluation Metrics**: Multi-Class Macro F1, Confusion Matrix Heatmaps.
- **Deliverables**: Batch processing pipeline analyzing live product review streams from public APIs.

### Project 6: Enterprise Knowledge Retrieval-Augmented Generation (RAG) System
- **Domain**: Generative AI & Enterprise Search
- **Core Techniques**: Recursive character text chunking with overlap, dense vector embeddings (OpenAI text-embedding-3 or BAAI/bge-m3), FAISS / Chroma vector database indexing, Reciprocal Rank Fusion (RRF) with BM25, Causal LLM response generation with citations.
- **Evaluation Metrics**: Context Precision, Context Recall, Faithfulness, and Answer Relevance (Ragas framework).
- **Deliverables**: Production-ready RAG assistant answering domain queries with strict hallucination safeguards.

---

## 🛠️ Recommended Tech Stack for Projects

```text
+-----------------------------+-------------------------------------------------+
| Layer                       | Recommended Tooling                             |
+-----------------------------+-------------------------------------------------+
| Data Processing             | Pandas, Polars, DuckDB, NumPy                   |
| Classical Modeling          | Scikit-Learn, XGBoost, LightGBM, CatBoost       |
| Deep Learning               | PyTorch, Torchvision, Hugging Face Transformers |
| Experiment Tracking         | MLflow, Weights & Biases (W&B)                  |
| Model Serving               | FastAPI, Uvicorn, ONNX Runtime                  |
| Interactive Prototyping     | Gradio, Streamlit                               |
| Containerization & Testing  | Docker, Docker Compose, Pytest, Ruff            |
+-----------------------------+-------------------------------------------------+
```

---

## 📝 Project Execution Protocol
1. Duplicate [`project-template.md`](project-template.md) into your project workspace.
2. Commit code incrementally with clean Git commit conventions (`feat:`, `fix:`, `docs:`).
3. Never commit raw multi-gigabyte datasets to Git; use `.gitignore` and cloud object storage (S3 / GCS).
4. Ensure that the test suite passes with `pytest tests/` prior to model deployment.

