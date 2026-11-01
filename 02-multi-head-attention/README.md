# Multi Head Attention

Four parallel self-attention heads (8 dimensions each) run side by side
and concatenated, followed by a single layer. The model is trained tiny-shakespeare character-level pipeline.

## Files
- `prepare_data.py` — tokenizer and batching (copied from
  single-attention-head).
- `multihead.py` — `Head` (unchanged from single-head version),
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

Larger batches give a cleaner gradient estimate and a modestly better
floor, but the improvement is small and plateaus — the real ceiling here
is the 8-character context window, not batch size. Four heads extract
more *kinds* of pattern from that window than one head could, but none
of them can see further back than 8 characters, so this won't improve
substantially until block_size increases or more layers are stacked.

Generated text shows clearer improvement than the loss number suggests:
more correctly-capitalized words, colons after name-shaped tokens,
recognizable short words.

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


