# single-attention-head

A from-scratch implementation of one scaled dot-product self-attention
head in PyTorch

## Files
- `prepare_data.py` — loads tiny-shakespeare, tokenizer, batching.

- `attention.py` — defines `Head` (one attention head with causal masking)
  and `SingleHeadModule` (token + positional embeddings, one attention
  head, linear readout), plus training and generation.

## Results
Trained for 5,000 steps, batch size 32-64, block size 8, n_embd 32, head
size 16. Loss settles in the 2.3-2.5 range - a modest but real improvement. A single attention head cannot probably capture a lot.

Sample generation after training:
```
Won wear dalt sorgares ken soug me busey tave mee ano be pit ishe
```

Not yet real words, but visibly different from bigram output. Some
correct short words and more plausible letter clusters, showing the
model using more than one character of context for the first time.

## How to run
```bash
python3 -m venv venv
source venv/bin/activate
pip install torch
python3 attention.py
```
