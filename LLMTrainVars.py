import torch

# batch_size = 64
# block_size = 256
# max_iters = 15000
# eval_iter = 500
# learning_rate = 3e-4
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# eval_iters = 200
# n_embed = 384
# head_size = 384
# train_size = 0.9
# num_heads = 6
# num_layer = 6
# drop_out = 0.2

batch_size = 32
block_size = 64
max_iters = 4000
eval_iter = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 64
head_size = n_embed
train_size = 0.9
num_heads = 4
num_layer = 6
drop_out = 0.2