import torch
x = torch.randn(3, 4)

dim1 = torch.stack((x, x), dim=0)
print(dim1)
print(dim1.size())

dim2 = torch.stack((x, x), dim=1)
print(dim2)
print(dim2.size())

dim3 = torch.stack((x, x), dim=2)
print(dim3)
print(dim3.size())