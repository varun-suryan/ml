import torch
from torch.nn import functional as F

b, t, c = 2, 3, 4

x = torch.randn(b, t, c)
print(x)
print(x[:, -1, :])