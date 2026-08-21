# RNN and LSTM

## Concept
Networks for sequential data that carry a hidden state through time.

## Formula / Core Idea
h_t = f(W x_t + U h_(t-1)). LSTM and GRU add gates to remember or forget.

## When to Use
Time series, older NLP tasks, speech.

## Example
Predict the next value in a stock or sensor sequence.

## Key Points
- Plain RNNs struggle with long dependencies
- LSTM gates control memory
- Largely replaced by Transformers in NLP
- Still useful for some time series

## My Notes
- (add your own notes here)
