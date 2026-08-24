# Text Preprocessing

## Concept
Cleaning and converting raw text into a form models can use.

## Formula / Core Idea
Lowercase, tokenise, remove stopwords, stem or lemmatise, vectorise (Bag of Words, TF-IDF).

## When to Use
Classical NLP such as sentiment and spam detection.

## Example
```python
from sklearn.feature_extraction.text import TfidfVectorizer
X = TfidfVectorizer().fit_transform(texts)
```

## Key Points
- Modern models use their own tokenizers
- TF-IDF is a strong simple baseline
- Do not over-clean for transformer models

## My Notes
- (add your own notes here)
