# Word & Sentence Embeddings: TF-IDF, Word2Vec & SBERT

## Concept
An **embedding** is a mapping of discrete categorical tokens (words, sentences, or documents) into a continuous,
dense vector space $\mathbb{R}^D$ where geometric proximity (distance and angle) reflects semantic similarity.

This foundation rests on the **Distributional Hypothesis** formulated by linguist J.R. Firth in 1957:
*"You shall know a word by the company it keeps."* Words that occur in similar linguistic contexts have similar meanings.
While classical sparse representations (TF-IDF) represent documents as orthogonal, multi-thousand dimensional vectors,
dense embeddings (Word2Vec, Sentence-BERT) compress semantic meaning into low-dimensional dense vectors ($D \in [384, 1536]$),
powering modern vector databases, semantic search engines, and Retrieval-Augmented Generation (RAG) architectures.

---

## Formula / Core Idea

### 1. Term Frequency-Inverse Document Frequency (TF-IDF)
For word $t$, document $d$, and corpus $D$ with $N = |D|$ total documents:
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

Where:
- $\text{TF}(t, d) = \frac{f_{t, d}}{\sum_{t' \in d} f_{t', d}}$ (Frequency of term in document).
- $\text{IDF}(t, D) = \ln\left( \frac{1 + N}{1 + |\{d \in D : t \in d\}|} \right) + 1$ (Penalizes ubiquitous words like "the", "is").

### 2. Word2Vec: Skip-Gram with Negative Sampling (SGNS)
Given center word $w_c$, predict context word $w_o$ within a sliding window of radius $R$.
Standard softmax is computationally intractable due to summing over vocabulary $V$ ($|V| > 100,000$):
$$P(w_o \mid w_c) = \frac{\exp(\mathbf{v}_{w_o}'^T \mathbf{v}_{w_c})}{\sum_{w \in V} \exp(\mathbf{v}_w'^T \mathbf{v}_{w_c})}$$

**Negative Sampling (Mikolov et al., 2013)** transforms this into binary logistic regression:
$$\mathcal{L}_{\text{SGNS}} = \log \sigma(\mathbf{v}_{w_o}'^T \mathbf{v}_{w_c}) + \sum_{k=1}^K \mathbb{E}_{w_n \sim P_n(w)} [\log \sigma(-\mathbf{v}_{w_n}'^T \mathbf{v}_{w_c})]$$
Where $K$ noise words $w_n$ are sampled according to the unigram distribution raised to the $3/4$ power.

### 3. Sentence Transformers & The Bi-Encoder Architecture
Standard BERT computes cross-attention over concatenated pairs, requiring $O(N^2)$ forward passes to find the nearest document.
**Sentence-BERT (SBERT; Reimers & Gurevych, 2019)** uses Siamese networks with Mean Pooling to produce fixed sentence embeddings $\mathbf{u}, \mathbf{v}$:
$$\text{Similarity}(\mathbf{u}, \mathbf{v}) = \cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u}^T \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$

---

## When to Use

- **TF-IDF & BM25**: Best for exact keyword search (acronyms, model part numbers, legal clause identifiers).
- **Dense Sentence Embeddings (SBERT / OpenAI text-embedding-3)**: Best for semantic search, concept matching, and RAG retrieval.
- **Hybrid Search (Sparse BM25 + Dense Vectors)**: The production gold standard for enterprise information retrieval systems.

---

## Example

Below is a complete implementation of a Vector Semantic Search engine using pure NumPy and Scikit-Learn TF-IDF:

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple

class VectorSearchEngine:
    """Information retrieval engine computing cosine similarity over dense/sparse embeddings."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_embeddings: np.ndarray = np.array([])
        self.documents: List[str] = []
        
    def index_corpus(self, documents: List[str]) -> None:
        self.documents = documents
        # Fit and transform documents to L2-normalized TF-IDF vectors
        sparse_matrix = self.vectorizer.fit_transform(documents)
        self.doc_embeddings = sparse_matrix.toarray()
        print(f"Indexed {len(documents)} documents with vocabulary size {self.doc_embeddings.shape[1]}.")
        
    def search(self, query: str, top_k: int = 3) -> List[Tuple[float, str]]:
        query_vec = self.vectorizer.transform([query]).toarray()
        # Compute Cosine Similarity (vectors already L2-normalized by TF-IDF)
        scores = (self.doc_embeddings @ query_vec.T).flatten()
        # Rank indices descending
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(float(scores[idx]), self.documents[idx]) for idx in top_indices]

if __name__ == "__main__":
    corpus = [
        "Convolutional neural networks are widely used in computer vision.",
        "Natural language processing models leverage attention and transformers.",
        "Gradient descent and backpropagation optimize deep learning weights.",
        "Support vector machines find maximum margin separation hyperplanes.",
        "Large language models generate text using autoregressive token sampling.",
    ]
    
    searcher = VectorSearchEngine()
    searcher.index_corpus(corpus)
    
    query = "attention transformer language"
    results = searcher.search(query, top_k=2)
    print(f"\nQuery: '{query}'")
    for score, doc in results:
        print(f"- Score: {score:.4f} | Document: {doc}")
```

---

## Key Points & Pitfalls

### 1. Vector Normalization (Unit Length)
Always normalize embedding vectors to unit length ($\mathbf{u} = \frac{\mathbf{u}}{\|\mathbf{u}\|_2}$).
When vectors have unit length, the Cosine Similarity simplifies to the simple dot product ($\mathbf{u}^T \mathbf{v}$),
allowing vector databases (FAISS, Milvus) to compute millions of similarity scores via hardware-accelerated BLAS matrix multiplication.

### 2. The Semantic Blindspot of Dense Retrieval
Dense embeddings struggle with specific keyword searches (e.g. "Error Code 0x80070057").
Combining dense retrieval with sparse BM25 keyword matching via Reciprocal Rank Fusion (RRF) avoids retrieval failures.

### 3. Mean Pooling vs [CLS] Token
For sentence embeddings with BERT-like models, Mean Pooling across all token output vectors significantly outperforms
using the raw `[CLS]` token representation.

---

#