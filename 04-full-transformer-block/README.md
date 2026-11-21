# Transformer with Stacked Blocks and LayerNorm


The full architecture: four transformer blocks stacked, with LayerNorm inside each block and one more before the output layer. Trained on tiny-shakespeare.

Compared  to the code of one block, there are three additions.

1. **LayerNorm inside every block:** 
   
	For each position separately, LayerNorm takes its 32 numbers, subtracts their mean, divides by their standard deviation, then applies a learned scale and shift (32 values each, starting at 1 and 0).
	It never mixes information between positions. That happens only in attention. The common motivation is to keep the scale of the numbers steady as many residual additions pile up (Ba et al., 2016). 
	It is applied like following
	
```python
	x = x + self.sa(self.ln1(x))
	x = x + self.ffwd(self.ln2(x))
```

2. **Four blocks stacked** instead of one.
3. **A final LayerNorm** before the output layer.

`Head`, `MultiHeadAttention`, `FeedForward` and the training loop are unchanged.


## Results


Trained for 5,000 steps, n_embd 32, 4 heads of size 8, 4 layers, block_size 8.



| batch_size | settled loss band |
| ---------- | ----------------- |
| 8          | ~2.1–2.4          |
| 16         | ~1.9–2.3          |
| 32         | ~1.9–2.1          |
| 64         | ~1.8–2.1          |


#### Sample output (batch size 64)

```
Sise seistain;
Mand well the gent all fulond not it the too,
Nor cound town flace for sio to assill all.

BENHORLIA:
For mise
banderer it mepul preck this will yeir briefors; and tratir I treing deating his And fiels, what no shoodly!
Shich bege actions:
A pronor nies?

EDWA:
Go?
To brother, gelious
```



## Another Experiment with higher block size


Trained for 5,000 steps, n_embd 32, 4 heads of size 8, 4 layers, and **block_size 32**.


| batch_size | settled loss band |
| ---------- | ----------------- |
| 8          | ~1.8-2.0          |
| 16         | ~1.8-2.0          |
| 32         | ~1.8–2.0          |
| 64         | ~1.7–1.9          |


#### Sample output (batch size 64)

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


#### Total Parameters

| part | count |
|---|---|
| token embedding (65 × 32) | 2,080 |
| position embedding (32 × 32) | 1,024 |
| one block: attention heads 3,072 + projection 1,056 + feed-forward 8,352 + two LayerNorms 128 | 12,608 |
| four blocks | 50,432 |
| final LayerNorm | 64 |
| output layer | 2,145 |
| **total** | **55,745** |

It can be a good experiment to change different type of parameters and see what is observed on the generated text.

## How to run

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch
python3 stacked.py
```

## Still different from GPT-2

Characters instead of byte-pair tokens, ReLU instead of GELU, no dropout, no weight tying, and attention scores scaled by 1/√n_embd instead of 1/√head_size.

