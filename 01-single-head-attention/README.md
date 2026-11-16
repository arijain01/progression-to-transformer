# Transformer consisting of exactly One Attention Head

Code for a single attention head. Each token looks back at earlier tokens and takes a weighted average of what they carry.

## Files
- `prepare_data.py` — loads tiny-shakespeare, tokenizer, batching.

- `attention.py` — defines `Head` (one attention head with causal masking)
  and `SingleHeadModule` (token + positional embeddings, one attention
  head, linear readout), plus training and generation.

## Results
Trained for 5,000 steps, batch size 64, block size 8, n_embd 32, head
size 16. Loss settles in the 2.3-2.5 range. With no training, the loss is around 4.3. Therefore, it is at least not randomly picking. A single attention head cannot probably capture a lot.

Sample generation of 300 tokens after training:

```
BO ETENNHAUEno Ant mbe:
ABLAnow ime hy theve's none wel! 't, our fal dilifoy,
Rafy mmest ornouepeayreeald:
AI farind ut aur ingeridu che hel at the norrepou I thelebarvo youes dlof gle; ade mes thin,
FUrd Ss on; omjoom:
TLOLUCO:
AROA hurid ceans it nd nof moul, ye, brotan ald cwher masn cked:
Th's-
```

There is some structure like colon, appostrophe in the output but the words do not make sense.

## How to run
```bash
python3 -m venv venv
source venv/bin/activate
pip install torch
python3 attention.py
```
