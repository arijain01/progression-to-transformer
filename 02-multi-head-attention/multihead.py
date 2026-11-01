import torch
import torch.nn as nn
import torch.nn.functional as F
from prepare_data import vocab_size, decode, get_batch

# We write the other values here again. Since we are doing mult-headed attention, we have split the head_size to 8 but have increased the number of heads
block_size = 8
n_embd = 32
n_head = 4 
head_size = n_embd//n_head


class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd,head_size,bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        v = self.value(x)
        out = wei@v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class MultiHeadModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_heads = MultiHeadAttention(n_head, head_size)
        self.lm_head = nn.Linear(n_embd,vocab_size)

    def forward(self,idx,targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T))
        x = tok_emb + pos_emb
        x = self.sa_heads(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def train_model(model, steps=5000):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(steps):
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb) #forward function is called here
        optimizer.zero_grad(set_to_none=True) #clearing out previous gradients
        loss.backward()
        optimizer.step()
        if step % 200 == 0:
            print(f"step {step}: loss {loss.item():.4f}")
    print(f"Final loss: {loss.item():.4f}")
    return model

if __name__ == "__main__":
    model = MultiHeadModel(vocab_size)
    model = train_model(model, steps=5000)
    idx = torch.zeros((1, 1), dtype=torch.long)
    print(decode(model.generate(idx, max_new_tokens=300)[0].tolist()))