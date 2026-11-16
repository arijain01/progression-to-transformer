#  Transformer with a Single Attention Head


Code for a single attention head. 

## Files

* `prepare_data.py` — loads tiny-shakespeare, builds the tokenizer, and handles batching.
* `attention.py` — defines `Head` (one attention head with causal masking) and `SingleHeadModule` (token and positional embeddings, one attention head, and a linear readout), plus training and generation.

## Results


Trained for 5,000 steps with batch size 64, block size 8, n_embd 32, and head size 16. The loss settles in the 2.3–2.5 range, down from about 4.3 before training. For comparison, a uniform random guesser over a 65-character vocabulary would score ln 65 ≈ 4.17, so the model is clearly learning something. 


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
