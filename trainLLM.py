import torch
import torch.nn as nn
from torch.nn import functional as F

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(li):
    return "".join([itos[i] for i in li])

data = torch.tensor(encode(text), dtype=torch.long)
train_size = 0.9
n = int(train_size*len(data))

train_data, val_data = data[:n], data[n:]

block_size = 8
print(train_data[:block_size+1])

x, y = train_data[:block_size], train_data[1:block_size+1]
for t in range(block_size):
    context, target = x[:t+1], y[t]
    print("When context is {}, the target is {}".format(context.tolist(), target.tolist()))

torch.manual_seed(1337)
batch_size = 4
def get_batch(dataType: str):
    _data = train_data if dataType == "train" else val_data
    ix = torch.randint(len(_data)-block_size, (batch_size, ))
    x = torch.stack([_data[i:i+block_size] for i in ix])
    y = torch.stack([_data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch("train")
print("inputs:")
print(xb.shape, yb.shape)
print(xb)
print(yb)

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
    def forward(self, idx, targets):
        logits = self.token_embedding_table(idx)
        return logits

m = BigramLanguageModel(vocab_size)
out = m(xb, yb)
print(out)