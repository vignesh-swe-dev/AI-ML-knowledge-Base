# Transformers

## Concept
Architecture based on self-attention, processing whole sequences in parallel.

## Formula / Core Idea
Attention(Q,K,V) = softmax(Q K^T / sqrt(d)) V

## When to Use
Language models, translation, vision transformers, almost all modern AI.

## Example
BERT (encoder), GPT (decoder), T5 (encoder-decoder).

## Key Points
- Self-attention relates every token to every other
- Positional encoding adds order
- Pretrain then fine-tune
- Scales very well with data and compute

## My Notes
- (add your own notes here)
