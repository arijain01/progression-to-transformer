# Transformer with Multi-Head Attention

Four parallel self-attention heads (8 dimensions each) run side by side, and their outputs are concatenated. The model is trained on tiny-shakespeare, using the same pipeline as the single-head stage.

## Files
- `prepare_data.py` — tokenizer and batching (copied from
  single-attention-head).
- `multihead.py` — contains `Head` (unchanged from single-head version),
  `MultiHeadAttention` (runs 4 heads in parallel via `nn.ModuleList`,
  concatenates their outputs, projects back to `n_embd` width), and
  `MultiHeadModel`.

## Results

Trained for 5,000 steps, block_size 8, n_embd 32, 4 heads of size 8 each.
Swept batch_size across 8/16/32/64 to separate real improvement from
training noise:

| batch_size | settled loss band |
|---|---|
| 8  | ~2.4-2.6 |
| 16 | ~2.3-2.5 |
| 32 | ~2.2-2.4 |
| 64 | ~2.1-2.3 |


Larger batches lower the settled loss and give a more stable curve. Part of this is simply that a larger batch sees more data in the same 5,000 steps. 

In this sample, the generated text looks like a clearer improvement than the loss alone suggests: more correctly capitalized words, colons after name-like tokens, and recognizable short words such as be, is, and as.


# Sample Output (Batch Size is 64) 

```
MES: Eden I of ith fevectesewt, the viag forowut hearckll be mars thas ath not a win path lid ce incoame, id is
E Fyeempit.

Hit fornor; he rut lof firend ifns fole, hy
Sut weit istis off yould nasize.

MEMA mpllioth?

'THome at lothiourds, swacthe pir is ath sthour ter.

KINcis as thone tih wilt s
```


## How to run
```bash
python3 -m venv venv
source venv/bin/activate
pip install torch
python3 multihead.py
```


