import torch
import torch.nn as nn


class L2Norm(nn.Module):
    def __init__(self, eps=1e-20):
        super(L2Norm, self).__init__()
        self.eps = eps

    def forward(self, input):
        norm = torch.linalg.norm(input, dim=1, keepdim=True) + self.eps
        input = input / norm
        return input
