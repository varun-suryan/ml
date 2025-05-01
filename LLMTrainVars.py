import torch

batch_size = 32
block_size = 8
max_iters = 3000
eval_iter = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 32
train_size = 0.9