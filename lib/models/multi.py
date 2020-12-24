import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import copy

from lib.blocks.norm import L2Norm
from lib.blocks import GeM, SPoC, MAC


class MultiNet(nn.Module):
    def __init__(self, root=None):
        super(MultiNet, self).__init__()
        self.backbone_base, self.backbone_layer1, self.backbone_layer2, self.backbone_layer3, self.backbone_layer4 = self.load_backbone(root=root)

        self.mac = nn.AdaptiveMaxPool2d(1)
        self.spoc = nn.AdaptiveAvgPool2d(1)
        self.gem = GeM()

        self.linear1 = nn.Linear(in_features=2048, out_features=256)
        self.linear2 = nn.Linear(in_features=2048, out_features=256)
        self.l2 = L2Norm()

        for m in self.backbone_base.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)
        for m in self.backbone_layer1.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)
        for m in self.backbone_layer2.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)
        for m in self.backbone_layer3.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)
        for m in self.backbone_layer4.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)
        self._init()

    def forward(self, x):
        x = self.backbone_base(x)
        x = self.backbone_layer1(x)
        x = self.backbone_layer2(x)
        x = self.backbone_layer3(x)
        x = self.backbone_layer4(x)

        x1 = self.mac(x).view(x.shape[0], -1) + \
                self.spoc(x).view(x.shape[0], -1) + \
                self.gem(x)
        x1 = self.l2(self.linear1(x1))

        x2 = self.mac(x).view(x.shape[0], -1) + \
                self.spoc(x).view(x.shape[0], -1) + \
                self.gem(x)
        x2 = self.l2(self.linear2(x2))

        x = torch.cat((x1, x2), dim=1)
        return x

    def forward_proxy(self, x):
        x = x.requires_grad_(True)

        x = self.backbone_base(x)
        x_layer1 = self.backbone_layer1(x)
        x_layer2 = self.backbone_layer2(x_layer1)
        x_layer3 = self.backbone_layer3(x_layer2)
        x_layer4 = self.backbone_layer4(x_layer3)

        x1 = self.mac(x_layer4).view(x_layer4.shape[0], -1) + \
                self.spoc(x_layer4).view(x_layer4.shape[0], -1) + \
                self.gem(x_layer4)
        x1 = self.l2(self.linear1(x1))

        x2 = self.mac(x_layer4).view(x_layer4.shape[0], -1) + \
                self.spoc(x_layer4).view(x_layer4.shape[0], -1) + \
                self.gem(x_layer4)
        x2 = self.l2(self.linear1(x2))

        return x1, x2, x_layer1, x_layer2, x_layer3, x_layer4

    def forward_cam(self, x):
        with torch.no_grad():
            x = self.backbone(x)

        feature1 = x.detach().clone().requires_grad_(True)
        feature2 = x.detach().clone().requires_grad_(True)

        embedding1 = self.mac(feature1).view(x.shape[0], -1) + \
                self.spoc(feature1).view(x.shape[0], -1) + \
                self.gem(feature1)
        embedding1 = self.l2(self.linear1(embedding1))

        embedding2 = self.mac(feature2).view(x.shape[0], -1) + \
                self.spoc(feature2).view(x.shape[0], -1) + \
                self.gem(feature2)
        embedding2 = self.l2(self.linear2(embedding2))

        return feature1, feature2, embedding1, embedding2

    def forward_feature(self, x):
        return self.forward(x)

    def freeze_bn(self):
        for m in self.backbone_base.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        for m in self.backbone_layer1.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        for m in self.backbone_layer2.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        for m in self.backbone_layer3.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        for m in self.backbone_layer4.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def requires_grad_backbone(self, val=False):
        for param in self.backbone_base.parameters():
            param.requires_grad = val
        for param in self.backbone_layer1.parameters():
            param.requires_grad = val
        for param in self.backbone_layer2.parameters():
            param.requires_grad = val
        for param in self.backbone_layer3.parameters():
            param.requires_grad = val
        for param in self.backbone_layer4.parameters():
            param.requires_grad = val

    def _init(self):
        nn.init.kaiming_normal_(self.linear1.weight, mode='fan_out')
        nn.init.constant_(self.linear1.bias, 0)
        #nn.init.kaiming_normal_(self.linear2.weight, mode='fan_out')
        #nn.init.constant_(self.linear2.bias, 0)

    def load_backbone(self, root=None):
        backbone = models.resnet50()
        if root is not None:
            backbone.load_state_dict(torch.load(root))

        backbone_base = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        backbone_layer1 = backbone.layer1
        backbone_layer2 = backbone.layer2
        backbone_layer3 = backbone.layer3
        backbone_layer4 = backbone.layer4

        return backbone_base, backbone_layer1, backbone_layer2, backbone_layer3, backbone_layer4
