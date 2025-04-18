import torch

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

