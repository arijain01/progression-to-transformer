# 04-full-transformer-block

Complete GPT-style architecture: token + position embeddings, 4 stacked
pre-norm transformer blocks (LayerNorm -> multi-head attention ->
residual, LayerNorm -> feed-forward -> residual), final LayerNorm,
linear readout. n_embd 32, 4 heads, 5,000 steps, batch size 64.

## What each addition does
- **LayerNorm (pre-norm):** normalizes the input to each sublayer, not
  the residual stream itself, so the stream stays a clean additive
  accumulator.
- **Stacking 4 blocks:** adds depth (later layers build on earlier
  ones). It does NOT extend context: every layer sees the same window.
  Only block_size does that.

## Results (loss averaged over 200 batches)

| block_size | train | val | gap |
|---|---|---|---|
| 8  | 1.890 | 2.009 | 0.119 |
| 16 | 1.790 | 1.946 | 0.157 |
| 32 | 1.729 | 1.873 | 0.144 |

Each doubling of context lowered val loss by about 0.065, with a steady
train/val gap, so longer context was not just overfitting. The
block_size 32 curve was still falling fastest at step 5000, so it is
probably under-trained. Single run per setting; seed sensitivity not
tested.

## How to run
```bash
python3 stacked.py
```