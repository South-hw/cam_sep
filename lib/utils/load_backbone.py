import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import copy

def load_resnet50(root, num=2):
    backbone = models.resnet50()
    backbone.layer3[0].conv2.stride = (1, 1)
    backbone.layer3[0].downsample[0].stride = (1, 1)

    if root is not None:
        backbone.load_state_dict(torch.load(root))

    backbone_ = nn.Sequential(
        backbone.conv1,
        backbone.bn1,
        backbone.relu,
        backbone.maxpool,
        backbone.layer1,
        backbone.layer2,
        backbone.layer3,
    )

    layer4s = nn.ModuleList()
    for i in range(num):
        layer4s.append(copy.deepcopy(backbone.layer4))

    return backbone_, layer4s
