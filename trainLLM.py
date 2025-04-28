import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

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
train_size = 0.9
n = int(train_size * len(data))

train_data, val_data = data[:n], data[n:]

block_size = 8
print(train_data[:block_size + 1])

x, y = train_data[:block_size], train_data[1:block_size + 1]
# for t in range(block_size):
#     context, target = x[:t + 1], y[t]
#     print("When context is {}, the target is {}".format(context.tolist(), target.tolist()))

torch.manual_seed(1337)
batch_size = 4


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


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        _logits = self.token_embedding_table(idx)
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


m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
start = torch.zeros((1, 1), dtype=torch.long)
# print(decode(m.generate(start, 10)[0].tolist()))
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size, epochs = 32, 10000

losses = []
for epoch in range(epochs):
    xb, yb = get_batch("train", 512)
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

plt.plot(losses)
plt.show()
print(decode(m.generate(start, 100)[0].tolist()))
