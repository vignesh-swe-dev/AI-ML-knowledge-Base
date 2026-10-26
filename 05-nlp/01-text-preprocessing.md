# Text Preprocessing, Subword Tokenization & BPE

## Concept
Machine learning algorithms operate on mathematical vectors, not raw Unicode text strings.
The fundamental objective of text preprocessing is to convert raw, unstructured textual characters into a clean,
standardized sequence of integer token identifiers that an embedding layer can process.

Early NLP relied on whitespace and punctuation splitting (word-level tokenization), resulting in catastrophic failures:
exploding vocabulary tables, inability to handle typos or compound words, and frequent Out-of-Vocabulary (`<UNK>`) errors.
Modern Large Language Models universally rely on **Subword Tokenization** algorithms—primarily **Byte-Pair Encoding (BPE)**,
**WordPiece**, and **SentencePiece**—which dynamically decompose text into variable-length subword chunks based on corpus frequencies.

---

## Formula / Core Idea

### 1. Unicode Normalization Standards
Text collected from diverse web sources often contains visually identical characters with differing byte sequences.
The Unicode Consortium defines four canonical normalization forms:
- **NFC (Canonical Decomposition, followed by Canonical Composition)**: Default standard for clean text storage.
- **NFD (Canonical Decomposition)**: Separates base characters from accent diacritics (e.g. "é" $\to$ "e" + "´").
- **NFKC / NFKD (Compatibility Decomposition)**: Replaces compatibility glyphs with standard equivalents (e.g. converts ligature "ﬁ" to "fi", or superscripts "²" to "2").

### 2. The Byte-Pair Encoding (BPE) Algorithm
BPE is a compression algorithm adapted for subword vocabulary construction (Sennrich et al., 2016):
1. **Initialization**: Define base vocabulary $V$ containing all individual characters in the training corpus.
   Tokenize words in the corpus into sequences of characters, appending an end-of-word marker (e.g. `</w>`).
2. **Count Frequencies**: Count the co-occurrence frequency of all adjacent symbol pairs in the tokenized corpus.
3. **Merge Most Frequent Pair**: Identify the pair $(A, B)$ with highest frequency, add the merged symbol $AB$ to vocabulary $V$,
   and replace all occurrences of $(A, B)$ in the corpus with $AB$.
4. **Iterate**: Repeat steps 2 and 3 until the vocabulary reaches the target size $K$ (e.g. 32,000 or 128,000 tokens).

```text
Corpus: {"low": 5, "lower": 2, "newest": 6, "widest": 3}
Iter 0: Vocabulary = {l, o, w, e, r, n, s, t, i, d}
Iter 1: Most frequent pair = ('e', 's') [appears 9 times in 'newest' and 'widest']
        Merge 'e' + 's' -> 'es'
        Vocabulary += {'es'}
Iter 2: Most frequent pair = ('es', 't') [appears 9 times]
        Merge 'es' + 't' -> 'est'
        Vocabulary += {'est'}
```

### 3. Comparison of Subword Algorithms

| Algorithm | Merge / Selection Criterion | Used By |
|---|---|---|
| **Byte-Pair Encoding (BPE)** | Maximum raw pair co-occurrence frequency | GPT-2, GPT-3, GPT-4, Llama 1/2/3 |
| **WordPiece** | Maximizes mutual information / likelihood of unigram language model | BERT, DistilBERT, Electra |
| **SentencePiece (Unigram)** | Starts with massive vocabulary and iteratively prunes tokens that minimize loss | T5, ALBERT, MarianMT |
| **Byte-Level BPE** | Operates over 256 raw bytes of UTF-8 encoding; 0 OOV tokens | RoBERTa, GPT-4, Mistral |

---

## When to Use

- Use **Byte-Level BPE** for general multilingual LLMs and code generation models where Unicode and arbitrary bytes appear.
- Use **SentencePiece** when processing languages without clear whitespace word boundaries (e.g., Japanese, Chinese, Thai).
- Avoid aggressive stopword removal and stemming in modern LLM pipelines; language models require grammatical syntax to infer meaning.

---

## Example

Below is a complete, self-contained Python implementation of the **Byte-Pair Encoding (BPE)** training algorithm from scratch:

```python
import collections
from typing import Dict, Tuple, List

def get_pair_frequencies(corpus: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
    """Counts adjacent symbol pair frequencies across word dictionary."""
    pairs = collections.defaultdict(int)
    for word_tuple, freq in corpus.items():
        for i in range(len(word_tuple) - 1):
            pair = (word_tuple[i], word_tuple[i + 1])
            pairs[pair] += freq
    return pairs

def merge_vocab_pair(pair: Tuple[str, str], corpus: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
    """Replaces target pair occurrences with merged symbol across all corpus tuples."""
    new_corpus = {}
    bigram = pair
    replacement = "".join(pair)
    
    for word_tuple, freq in corpus.items():
        new_word = []
        i = 0
        while i < len(word_tuple):
            if i < len(word_tuple) - 1 and (word_tuple[i], word_tuple[i + 1]) == bigram:
                new_word.append(replacement)
                i += 2
            else:
                new_word.append(word_tuple[i])
                i += 1
        new_corpus[tuple(new_word)] = freq
    return new_corpus

def train_toy_bpe(raw_words: List[str], num_merges: int = 5) -> Tuple[List[str], Dict[Tuple[str, str], str]]:
    """Trains a basic BPE tokenizer, learning merge rules."""
    # Initialize corpus with characters split and end-of-word sentinel
    corpus_counts = collections.Counter(raw_words)
    corpus = {tuple(list(word) + ["</w>"]): count for word, count in corpus_counts.items()}
    
    merge_rules = {}
    for step in range(num_merges):
        pairs = get_pair_frequencies(corpus)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        merged_token = "".join(best_pair)
        merge_rules[best_pair] = merged_token
        corpus = merge_vocab_pair(best_pair, corpus)
        print(f"Merge #{step+1}: {best_pair} -> '{merged_token}' (Freq: {pairs[best_pair]})")
        
    return list(corpus.keys()), merge_rules

if __name__ == "__main__":
    training_text = ["low"] * 5 + ["lower"] * 2 + ["newest"] * 6 + ["widest"] * 3
    final_corpus, rules = train_toy_bpe(training_text, num_merges=4)
    print("\nLearned Merge Rules:", rules)
```

---

## Key Points & Pitfalls

### 1. Tokenizer Boundary Mismatches
Subword tokenizers treat leading spaces as distinct tokens (e.g. `" hello"` $\neq$ `"hello"`). In GPT tokenizers,
leading whitespace is denoted by `Ġ` (Unicode `U+0120`). Failing to strip or format whitespace identically during inference
causes models to generate garbled continuations.

### 2. Don't Stem or Lemmatize for Modern LLMs
Traditional techniques (Porter Stemmer) cut words to artificial roots (e.g. "running" $\to$ "run"). Modern Transformers
rely on exact verb tense and morphological suffixes to infer nuanced temporal and syntactic relationships.

### 3. Special Tokens Preservation
Always reserve special tokens (`<pad>`, `<unk>`, `<s>`, `</s>`, `[CLS]`, `[SEP]`, `<|im_start|>`) so they are never
fragmented by subword splits during tokenization.

---
