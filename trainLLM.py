import torch
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

print(train_data[:block_size + 1])

x, y = train_data[:block_size], train_data[1:block_size + 1]
# for t in range(block_size):
#     context, target = x[:t + 1], y[t]
#     print("When context is {}, the target is {}".format(context.tolist(), target.tolist()))

torch.manual_seed(1337)
def get_batch(dataType: str, batch_size=32):
    _data = train_data if dataType == "train" else val_data
    ix = torch.randint(len(_data) - block_size, (batch_size,))
    x = torch.stack([_data[i:i + block_size] for i in ix])
    y = torch.stack([_data[i + 1:i + block_size + 1] for i in ix])
    return x, y


xb, yb = get_batch("train")
# print("inputs:")
# print(xb.shape, yb.shape)
# print(xb)
# print(yb)

print('-----')
# for b in range(batch_size):
#     for t in range(block_size):
#         context, target = xb[b, :t+1], yb[b, t]
#         print(f"When input is {context.tolist()}, the target is {target}")

vocab_size = len(chars)


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embed, self.head_size, bias=False)
        self.query = nn.Linear(n_embed, self.head_size, bias=False)
        self.value = nn.Linear(n_embed, self.head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        b, t, c = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x) #(b, t, self.head_size)
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5
        wei = wei.masked_fill(self.tril[:t, :t] == 0, -float('inf')) #(b, t, t)
        wei = F.softmax(wei, dim=1)
        out = wei @ v
        return out # (b, t, self.head_size)



class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        tok_embed = self.token_embedding_table(idx) #(b, t, n_embed)
        pos_emb = self.position_embedding_table(torch.arange(t, device=device)) #(t, n_embed)
        x = tok_embed + pos_emb #(b, t, n_embed)

        _logits = self.lm_head(x) #(b, t, vocab_size=c)
        b, t, c = _logits.shape
        _logits = _logits.view(b*t, c)
        if targets is not None:
            targets = targets.view(b*t)
            _loss = F.cross_entropy(_logits, targets)
        else:
            _loss = None
        return _logits, _loss

    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            b, t = idx.shape
            _logits, _loss = self(idx)
            _logits = _logits[t - 1::t]  #(b, c) Just get the last time dimension
            probs = F.softmax(_logits, dim=1)  #(b,c)
            id_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, id_next), dim=1)
        return idx


m = BigramLanguageModel()
m = m.to(device)

logits, loss = m(xb, yb)
start = torch.zeros((1, 1), dtype=torch.long)
# print(decode(m.generate(start, 10)[0].tolist()))
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for name, param in m.named_parameters():
        print(name, param.data, param.shape)

@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for dType in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(dType)
            logits, loss = m(xb, yb)
            losses[k] = loss
        out[dType] = losses.mean()
    return out

# losses = []
for iter in range(max_iters):

    if iter % eval_iter == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch("train")
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    # losses.append(loss.item())

# plt.plot(losses)
# plt.show()
print(decode(m.generate(start, 100)[0].tolist()))
