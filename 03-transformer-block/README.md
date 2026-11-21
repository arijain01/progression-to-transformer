# Transformer Block: Feed-Forward Network and Residual Connections

One transformer block: multi-head attention followed by a feed-forward network, each wrapped in a residual connection. The model is trained on tiny-shakespeare, using the same pipeline as the earlier stages.

## Files

- `prepare_data.py` — tokenizer and batching (unchanged from the earlier stages).
- `block.py` — contains `Head` and `MultiHeadAttention` (unchanged from the multi-head stage), `FeedForward` (linear 32 → 128, ReLU, linear 128 → 32), `Block` (`x = x + attention(x)`, then `x = x + feedforward(x)`), and `BlockModel`, plus training and generation.


## Results

Trained for 5,000 steps with block_size 8, n_embd 32, and 4 heads of size 8 each. Swept batch_size across 8/16/32/64. The settled loss band is the range of the loss values printed from step 2,000 to step 4,800. Each printed value is the loss on a single mini-batch.

| batch_size | settled loss band |
| ---------- | ----------------- |
| 8          | ~2.1–2.6          |
| 16         | ~2.1–2.4          |
| 32         | ~2.1–2.4          |
| 64         | ~2.0–2.3          |

As seen previously, larger batches give a narrower band. Part of this is simply that a larger batch sees more data in the same 5,000 steps.

In this sample, the output has speaker-style tags (`QULEN:`), line breaks, punctuation, and several short real words (and, with, our, that, so, his), but most longer words are still invented.

## Sample output (batch size 64)

```
Ser now hit? light mie?

QULEN:
And weand with ingsetse cearbbas our the spele, Sishis., Gy trod.

PEERLARD Cont of my sor und,
Los!
Nor: I so is I mell,
Thou braveny. Hand insens!
Hall thingivel sull,
And hes, thal,
To wh that so his? willitcils spers to is braid!

For but nowe me liseck and mace a
```

## How to run

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch
python3 block.py
```
