import torch
from torch.nn import functional as F

b, t, c = 2, 2, 8

x = torch.randn(b, t, c)
print(x)
tril = torch.tril(torch.ones(t, t))
print(tril)
wei = torch.zeros((t, t))
wei = wei.masked_fill(tril==0, -float('inf'))
print(wei)
wei = F.softmax(wei, dim=1)
print(wei)
out = wei @ x
print(out.shape)
print(out)
