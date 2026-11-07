# 04-full-transformer-block

Complete GPT-style architecture: token + position embeddings, 4 stacked
pre-norm transformer blocks (LayerNorm -> multi-head attention ->
residual, LayerNorm -> feed-forward -> residual), final LayerNorm,
linear readout. n_embd 32, 4 heads, block_size 32, batch size 64,
5,000 steps. 55,745 parameters.

Builds on 03-transformer-block. New: stacking blocks, a final LayerNorm
(`ln_f`), and pre-norm.

## What each addition does
- **Stacking 4 blocks:** adds depth (later layers build on earlier
  ones). It does NOT extend context: every layer sees the same window.
  Only block_size increase context.
- **Final LayerNorm (`ln_f`):** normalizes the residual stream once
  more before the linear readout.
- **Pre-norm:** LayerNorm is applied to the input of each sublayer, not
  to the residual stream, so the stream stays a clean additive
  accumulator.

## Shapes
Every block maps batch_size x number_of_Tokens x n_embd to the same shape. Only the
readout changes it, to batch_size x number_of_Tokens x vocab_size.

## Result
Final training loss (last batch): 1.7100

Sample (300 characters):

```
FERCUS:
Have the her wound heald look of Corrow
more on the I love worldchman,
Far love unfiss,, in
Be may ittrious un!

RATICHAP:
She, canst no new heart, it fist
To are than to king, be that lut of why,
TRer plorsss; over I signs, whoms,
Alen you, disprinest's home;
Let whomans on lover
Do his ha
```


## How to run
```bash
python3 stacked.py
```
