# Module 05: Natural Language Processing & Large Language Models

## 📌 Module Overview
Natural Language Processing (NLP) is the branch of artificial intelligence dedicated to enabling computational systems
to analyze, represent, comprehend, and generate human language. Human language is inherently discrete, ambiguous,
hierarchical, and grounded in complex cultural common sense.

Over the past decade, NLP has undergone a complete paradigm revolution: transitioning from hand-crafted statistical rules
and n-gram language models to distributed word representations (Word2Vec), bidirectional neural encoders (BERT),
and finally to massive autoregressive Foundation Large Language Models (LLMs such as GPT-4, Llama 3, and Mistral).

This module covers the modern language engineering curriculum from first principles: subword tokenization algorithms,
dense vector representations, semantic vector retrieval, and autoregressive generation dynamics.

---

## 📚 Syllabus & Topic Matrix

| File | Subject | Core Focus | Engineering Application |
|---|---|---|---|
| [01-text-preprocessing.md](01-text-preprocessing.md) | Tokenization & Normalization | BPE, WordPiece, SentencePiece, Regex | Vocabulary building, tokenizer training |
| [02-embeddings.md](02-embeddings.md) | Dense Vector Embeddings | TF-IDF, Word2Vec, SBERT, Cosine Metric | Semantic search, RAG pipelines, clustering |
| [03-llms-basics.md](03-llms-basics.md) | Foundation LLM Architectures | Autoregressive decoding, Top-P, LoRA, RLHF | Instruction tuning, prompt engineering, generation |

---

## 🏛️ Evolution of NLP Paradigms

```text
1. Classical Statistical NLP (1990s - 2012):
   - Bag of Words (BoW), TF-IDF, N-gram Language Models, Hidden Markov Models (HMMs)
   - Limitation: Sparse, orthogonal representations; zero semantic relationship between 'dog' and 'canine'

2. Static Distributed Embeddings (2013 - 2017):
   - Word2Vec (Skip-Gram, CBOW), GloVe, FastText
   - Breakthrough: Continuous dense vectors capturing semantic analogies ('king' - 'man' + 'woman' = 'queen')
   - Limitation: Polysemy failure (the word 'bank' has one single vector regardless of financial or river context)

3. Contextual Pretrained Encoders (2018 - 2021):
   - ELMo, BERT, RoBERTa, DeBERTa
   - Breakthrough: Bidirectional self-attention yields dynamic, context-dependent token vectors

4. Generative Foundation Models & LLMs (2022 - Present):
   - GPT-4, Llama 3, Claude, Gemini
   - Breakthrough: Few-shot in-context learning, instruction tuning, multi-step chain-of-thought reasoning
```

---

## 💡 Top 5 Technical Interview Questions (NLP & LLMs)

### Q1: How does Byte-Pair Encoding (BPE) solve the Out-of-Vocabulary (OOV) problem?
**Answer**: Classical word-level tokenization assigns unique IDs to whole words. Any word unseen during training is
mapped to an uninformative `<UNK>` token. BPE (Sennrich et al., 2016) starts with a base vocabulary of individual characters
and iteratively merges the most frequent adjacent symbol pairs in the corpus. Rare words are decomposed into familiar
subword chunks (e.g. "unpredictable" $\to$ ["un", "predict", "able"]). In Byte-level BPE (used in GPT-2/3/4 and Llama),
the base vocabulary consists of the 256 raw bytes of UTF-8, guaranteeing that **no text string ever encounters an OOV token**.

### Q2: What is the difference between Top-K sampling and Top-P (Nucleus) sampling in LLM text generation?
**Answer**:
- **Top-K Sampling**: Truncates the probability distribution to a fixed number $K$ of candidate tokens with the highest probabilities.
  Problem: If the probability distribution is flat (many plausible words), $K$ might be too restrictive. If the distribution is
  extremely peaked (one obvious word), $K$ includes implausible tail tokens.
- **Top-P (Nucleus) Sampling**: Dynamically selects the smallest subset of tokens whose cumulative probability mass exceeds threshold $P$
  (e.g., $P = 0.90$). The candidate pool dynamically expands when uncertainty is high and contracts to a single token when confidence is high.

### Q3: What is the KV-Cache in LLM inference, and why is it essential?
**Answer**: In autoregressive generation, generating each new token requires computing attention over all preceding tokens.
Without caching, key ($K$) and value ($V$) vectors for past tokens would be recomputed from scratch at every generation step,
incurring $O(N^2)$ redundant matrix multiplications. The **KV-Cache** stores the computed $K$ and $V$ projection tensors of historical
tokens in GPU memory, allowing generation of token $N+1$ to execute in $O(N)$ time by multiplying the new single query against cached tensors.

### Q4: How does Low-Rank Adaptation (LoRA) enable efficient fine-tuning of 70B parameter models?
**Answer**: Full parameter fine-tuning updates all weights $W_0 \in \mathbb{R}^{d \times k}$, requiring storing optimizer moments for every parameter
(costing $16+$ bytes per parameter in AdamW). LoRA (Hu et al., 2021) freezes $W_0$ and parameterizes the weight update matrix $\Delta W$
as the low-rank product of two small matrices: $\Delta W = B \cdot A$, where $A \in \mathbb{R}^{r \times k}$, $B \in \mathbb{R}^{d \times r}$,
and rank $r \ll \min(d, k)$ (e.g. $r = 8$ or $16$). This reduces trainable parameters by **99%**, slashing GPU VRAM requirements.

### Q5: What is the difference between Bi-Encoders and Cross-Encoders in semantic search?
**Answer**:
- **Bi-Encoder (Sentence Transformers)**: Encodes query $q$ and document $d$ independently into fixed-dimensional vectors $\mathbf{u}, \mathbf{v}$.
  Similarity is computed via dot product or cosine distance. Millions of document vectors can be precomputed and indexed in vector databases
  for sub-millisecond approximate nearest neighbor search (MIPS / HNSW).
- **Cross-Encoder**: Passes query and document concatenated together (`[CLS] query [SEP] doc [SEP]`) through all Transformer attention layers.
  Cross-attention directly compares every query token with every document token, delivering significantly higher ranking accuracy
  at the cost of orders of magnitude higher computational latency. Standard pattern: Bi-encoder retrieves top 100 candidates; Cross-encoder re-ranks top 10.

---

## 📝 Study Checklist & Milestones
- [ ] Train a custom subword tokenizer using Hugging Face `tokenizers` library.
- [ ] Implement TF-IDF vectorization and cosine similarity search from scratch in pure NumPy.
- [ ] Build a vector retrieval pipeline using Sentence-Transformers and FAISS.
- [ ] Implement nucleus (top-p) sampling and temperature scaling from raw model logits.
- [ ] Fine-tune an open-source LLM (Llama / Mistral) using QLoRA on a custom instruction dataset.
