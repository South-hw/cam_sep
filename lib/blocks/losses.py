import torch
import torch.nn as nn
import torch.nn.functional as F


class AuxClsLoss(nn.Module):
    def __init__(self, cls, label_smooting=0.2, temperature_scaling=0.5):
        super(AuxClsLoss, self).__init__()
        self.confidence = 1.0 - label_smooting
        self.scaling = temperature_scaling
        self.cls = cls
        self.smooting = label_smooting

    def forward(self, pred, target):
        pred = F.log_softmax(pred / self.scaling, dim=1)

        with torch.no_grad():
            dist = torch.zeros_like(pred)
            dist.fill_(self.smooting / (self.cls - 1))
            dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-dist * pred, dim=1))
