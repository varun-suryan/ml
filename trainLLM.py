import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from LLMTrainVars import *

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]


def decode(li):
    return "".join([itos[i] for i in li])


data = torch.tensor(encode(text), dtype=torch.long)
n = int(train_size * len(data))

train_data, val_data = data[:n], data[n:]

torch.manual_seed(1337)


def get_batch(dataType: str, batch_size=32):
    _data = train_data if dataType == "train" else val_data
    ix = torch.randint(len(_data) - block_size, (batch_size,))
    x = torch.stack([_data[i:i + block_size] for i in ix])
    y = torch.stack([_data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


vocab_size = len(chars)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embed, self.head_size, bias=False)
        self.query = nn.Linear(n_embed, self.head_size, bias=False)
        self.value = nn.Linear(n_embed, self.head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.droput = nn.Dropout(drop_out)

    def forward(self, x):
        b, t, c = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)  #(b, t, self.head_size)
        wei = q @ k.transpose(-2, -1) * c ** -0.5  #(b, t, t). Probabably should be self.head_size**2
        wei = wei.masked_fill(self.tril[:t, :t] == 0, -float('inf'))  #(b, t, t)
        wei = F.softmax(wei, dim=-1)
        wei = self.droput(wei)
        out = wei @ v
        return out  # (b, t, self.head_size)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.head_size, self.num_heads = head_size, num_heads
        self.heads = nn.ModuleList([Head(self.head_size) for _ in range(self.num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embed)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class Feedforward(nn.Module):
    def __init__(self, n_input):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(drop_out)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.sa_heads = MultiHeadAttention(num_heads, head_size // num_heads)
        self.ffwd = Feedforward(head_size)
        self.ln1, self.ln2 = nn.LayerNorm(n_embed), nn.LayerNorm(n_embed)

    def forward(self, x):  # x: (b, t, n_embed)
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x  #(b, t, n_embed)


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(num_heads, head_size) for _ in range(num_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        tok_embed = self.token_embedding_table(idx)  #(b, t, n_embed)
        pos_emb = self.position_embedding_table(torch.arange(t, device=device))  #(t, n_embed)
        x = tok_embed + pos_emb  #(b, t, n_embed)
        x = self.blocks(x)  #(b, t, n_embed)
        x = self.ln_f(x)
        _logits = self.lm_head(x)  #(b, t, vocab_size=c)
        if targets is not None:
            b, t, c = _logits.shape
            _logits = _logits.view(b*t, c)
            targets = targets.view(b*t)
            _loss = F.cross_entropy(_logits, targets)
        else:
            _loss = None
        return _logits, _loss

    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            id_extracted = idx[:, -block_size:]
            _logits, _loss = self(id_extracted)  # crop the context for positional embedding table
            _logits = _logits[:, -1, :]  #(b, c) Just get the last time dimension. Might be buggy.
            probs = F.softmax(_logits, dim=1)  #(b,c)
            id_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, id_next), dim=1)
        return idx



@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for dType in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(dType, batch_size)
            logits, loss = m(xb, yb)
            losses[k] = loss.item()
        out[dType] = losses.mean()
    m.train()
    return out

m = BigramLanguageModel()
m = m.to(device)
print(device)
start = torch.zeros((1, 1), dtype=torch.long, device=device)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

losses_plot = []
for iter_ in range(max_iters):

    if iter_ % eval_iter == 0:
        losses = estimate_loss()
        print(f"Step {iter_}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch("train", batch_size)
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    losses_plot.append(loss.item())

# plt.plot(losses_plot)
# plt.show()
print(decode(m.generate(start, 1000)[0].tolist()))
# print(decode(m.generate(start, 100)[1].tolist()))
