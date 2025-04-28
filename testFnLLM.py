import torch
from torch.nn import functional as F

x = torch.tensor([[1., 2., 3.], [0., 0., 1.]])
print(F.softmax(x, dim=0))
print(F.softmax(x, dim=1))