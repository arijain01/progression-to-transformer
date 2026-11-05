# mini-gpt-journey

A GPT-style transformer in PyTorch,
following roughly Andrej Karpathy's "Let's build GPT" video. We do a step by step construction of the GPT so that the progression to GPT is visible. Built as proof-of-work for a pivot toward mechanistic interpretability.

## The journey

| Stage | Folder | What it adds | Loss on tiny-shakespeare |
|---|---|---|---|
| 1 | [`01-single-head-attention`](./01-single-head-attention) | A self-attention head | ~2.3–2.5 |
| 2 | [`02-multi-head-attention`](./02-multi-head-attention) | 4 parallel heads, concatenated + projected | ~2.1–2.3 |
| 3 | [`03-transformer-block`](./03-transformer-block) | Feed-forward network + residual connections | ~2.0–2.2 (dipped to 1.99) |
 4 | [`04-full-transformer-block`](./04-full-transformer-block) | Pre-Norm + LayerNorm | ~2.0–2.2 (dipped to 1.99) |

Each folder has its own README with the full results, sample generated
text, and how to run it. Each stage adds one real
architectural idea, and the loss (and the readability of the generated
Shakespeare) improves. Broadly, the improvement can be traced to the new idea.


## How to run any stage

```bash
cd 0X-stage-name
python3 -m venv venv
source venv/bin/activate
pip install -r ../requirements.txt
python3 <script>.py
```