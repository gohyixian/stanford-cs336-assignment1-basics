import torch
from cs336_basics.lm.arch.layers.activations import softmax


inf = torch.Tensor([float("-inf")])
out = softmax(inf)
print(out)