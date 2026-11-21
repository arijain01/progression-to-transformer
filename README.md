# Progression to Transformer-Style Architecture

Four small PyTorch programs that build from a single attention head up to a
GPT-style transformer, one major idea per stage, trained on tiny-shakespeare.
The aim is to make the progression visible.

## Credit and AI disclosure
Inspired by Andrej Karpathy's "Let's build GPT" video; this is not a
reproduction of his code. I built the stages with substantial help from
Claude (Anthropic) in planning and coding. I typed the code, ran, debugged, and wrote the explanations for each stage as a way to
learn the architecture.

## The Progression

| Stage | Folder | What it adds | Loss on tiny-shakespeare |
|---|---|---|---|
| 1 | [`01-single-head-attention`](./01-single-head-attention) | A self-attention head | ~2.3–2.5 |
| 2 | [`02-multi-head-attention`](./02-multi-head-attention) | 4 parallel heads, concatenated + projected | ~2.1–2.3 |
| 3 | [`03-transformer-block`](./03-transformer-block) | Feed-forward network + residual connections | ~2.0–2.2  |
| 4 | [`04-full-transformer-block`](./04-full-transformer-block) | LayerNorm | ~2.0–2.2 (to be checked) |

Each folder has its own README with the full results, sample generated
text, and how to run it. Each stage adds one major
architectural idea. Loss generally falls as stages are added, but these are single runs on a tiny model, so small differences between stages are within noise.


## How to run any stage

```bash
cd 0X-stage-name
python3 -m venv venv
source venv/bin/activate
pip install -r ../requirements.txt
python3 <script>.py
```
