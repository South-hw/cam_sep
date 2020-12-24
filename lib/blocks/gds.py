import torch
import torch.nn as nn


class MAC(nn.Module):
    def __init__(self, in_channels, out_channels, num_cls=None):
        super(MAC, self).__init__()
        self.linear = nn.Linear(in_features=in_channels,
                out_features=out_channels, bias=True)
        self.num_cls = num_cls
        if num_cls is not None:
            self.cls_layer = nn.Linear(in_features=2048, out_features=num_cls,
                    bias=True)
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, -1)
        x = x.max(dim=2)[0]
        
        cls_out = None
        if self.num_cls is not None:
            cls_out = self.cls_layer(x)
            x = self.linear(x)

            return x, cls_out
        else:
            x = self.linear(x)
            return x


class SPoC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SPoC, self).__init__()
        self.linear = nn.Linear(in_features=in_channels,
                out_features=out_channels, bias=True)

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.sum(dim=(2, 3)) / (h * w)
        x = self.linear(x)
        return x

class GeM(nn.Module):
    def __init__(self, p=3, learnable=False):
        super(GeM, self).__init__()

        if learnable:
            self.p = nn.Parameter(torch.tensor(3.0))
        else:
            self.p = 3.0


    def forward(self, x):
        b, c, h, w = x.shape

        x = x.pow(self.p).sum(dim=(2, 3)) / (h * w)
        x = x + 1e-20
        x = x.pow(1 / self.p)

        return x


