import torch
import torch.nn as nn
import torchvision
import torchvision.models as models

from lib.blocks.norm import L2Norm
from lib.blocks import GeM, SPoC, MAC

class BaseNet(nn.Module):
    def __init__(self, root=None):
        super(BaseNet, self).__init__()
        self.backbone = self.load_backbone(root=root)

        self.mac1 = nn.AdaptiveMaxPool2d(1)
        self.spoc1 = nn.AdaptiveAvgPool2d(1)
        self.gem1 = GeM()
        self.linear1 = nn.Linear(in_features=2048, out_features=256)

        self.mac2 = nn.AdaptiveMaxPool2d(1)
        self.spoc2 = nn.AdaptiveAvgPool2d(1)
        self.gem2 = GeM()
        self.linear2 = nn.Linear(in_features=2048, out_features=256)

        self.l2 = L2Norm()

        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)
        self._init()

    def _init(self):
        nn.init.kaiming_normal_(self.linear1.weight, mode='fan_out')
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.kaiming_normal_(self.linear2.weight, mode='fan_out')
        nn.init.constant_(self.linear2.bias, 0)

    def forward(self, x):
        x = self.backbone(x)

        mac_x1 = self.mac1(x).view(x.shape[0], -1)
        spoc_x1 = self.spoc1(x).view(x.shape[0], -1)
        gem_x1 = self.gem1(x)

        x1 = mac_x1 + gem_x1 + spoc_x1
        x1 = self.linear1(x1)
        x1 = self.l2(x1)

        mac_x2 = self.mac2(x).view(x.shape[0], -1)
        spoc_x2 = self.spoc2(x).view(x.shape[0], -1)
        gem_x2 = self.gem2(x)

        x2 = mac_x2 + gem_x2 + spoc_x2
        x2 = self.linear2(x2)
        x2 = self.l2(x2)

        x = torch.cat((x1, x2), dim=1)
        return x

    def forward_feature(self, x):
        return self.forward(x)

    def load_backbone(self, root=None):
        backbone = models.resnet50()

        if root is not None:
            backbone.load_state_dict(torch.load(root))
        backbone = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
                backbone.layer4
        )
        return backbone

    def freeze_bn(self):
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def requires_grad_backbone(self, val=False):
        for param in self.backbone.parameters():
            param.requires_grad = val
