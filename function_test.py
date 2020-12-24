import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.blocks import GlobalDescriptor
from lib.models import MultiNet

net = MultiNet(root=None, num_desc=1)
net.train()

x = torch.rand(size=(3, 3, 224, 224))
y = net.forward_proxy(x)

pdb.set_trace()
print(y.shape)
