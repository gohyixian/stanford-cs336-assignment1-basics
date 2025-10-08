import torch
from cs336_basics.model.layers.activations import softmax


inf = torch.Tensor([float("-inf")])
out = softmax(inf)
print(out)