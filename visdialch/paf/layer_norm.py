import torch
import torch.nn as nn
from anatool import AnaLogger


class LayerNorm(nn.Module):
    def __init__(self, logger: AnaLogger, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.logger = logger
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
