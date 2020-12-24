import os
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import grad
import torchvision.transforms as transforms

from imgcat import imgcat
import numpy as np


from pytorch_metric_learning.losses import ProxyAnchorLoss
from pytorch_metric_learning.samplers import MPerClassSampler
from lib.datasets import CUB2011
from lib.models import MultiNet

import cv2
from PIL import Image
import pdb

img_root = '/home/nam/research/cam/outputs/imgs'

def cam(net, loss_fn1, loss_fn2, dl):

    iter_dl = iter(dl)
    imgs, cls = next(iter_dl)

    imgs = imgs.cuda()
    cls = cls.cuda()
    
    feature1, feature2, embedding1, embedding2 = net.forward_cam(x=imgs)

    loss1 = loss_fn1(embedding1, cls)
    loss2 = loss_fn2(embedding2, cls)
    
    b, _, h, w = feature1.shape
    alpha1 = grad(outputs=loss1, inputs=feature1, retain_graph=True,
            create_graph=True)[0]
    alpha1 = F.relu(alpha1).sum(dim=(2, 3))
    cam1 = alpha1[:, :, None, None] * feature1
    cam1 = F.relu(cam1.sum(dim=1)) / (h * w)
    cam1 = F.upsample(cam1.unsqueeze(1), size=(imgs.shape[2], imgs.shape[3]), mode='bilinear')


    alpha2 = grad(outputs=loss2, inputs=feature2, retain_graph=True,
            create_graph=True)[0]
    alpha2 = F.relu(alpha2).sum(dim=(2, 3))
    cam2 = alpha2[:, :, None, None] * feature2
    cam2 = F.relu(cam2.sum(dim=1)) / (h * w)
    cam2 = F.upsample(cam2.unsqueeze(1), size=(imgs.shape[2], imgs.shape[3]), mode='bilinear')

    examples = range(128)
    mean = torch.tensor([0.485, 0.456, 0.406]).float().cuda().view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).float().cuda().view(3, 1, 1)
    for e in examples:
        heatmap1 = cam1[e]
        heatmap1 = heatmap1 - torch.min(heatmap1)
        heatmap1 = heatmap1 / torch.max(heatmap1)
        heatmap1 = heatmap1.detach().cpu().numpy().transpose(1, 2, 0)
        heatmap1 = cv2.applyColorMap(np.uint8(255 * heatmap1), cv2.COLORMAP_JET)
        heatmap1 = np.float32(heatmap1) / 255
        heatmap1 = heatmap1[:, :, ::-1]

        heatmap2 = cam2[e]
        heatmap2 = heatmap2 - torch.min(heatmap2)
        heatmap2 = heatmap2 / torch.max(heatmap2)
        heatmap2 = heatmap2.detach().cpu().numpy().transpose(1, 2, 0)
        heatmap2 = cv2.applyColorMap(np.uint8(255 * heatmap2), cv2.COLORMAP_JET)
        heatmap2 = np.float32(heatmap2) / 255
        heatmap2 = heatmap2[:, :, ::-1]

        img = imgs[e] * std + mean
        img = img.detach().cpu().numpy().transpose(1, 2, 0)

        result1 = heatmap1 + img
        result1 = result1 / np.max(result1)
        result2 = heatmap2 + img
        result2 = result2 / np.max(result2)

        print(result1.shape)
        result1 = Image.fromarray((result1*255).astype(np.uint8))
        result1.save(os.path.join(img_root, f'{e}_cub1.jpeg'))
        result2 = Image.fromarray((result2*255).astype(np.uint8))
        result2.save(os.path.join(img_root, f'{e}_cub2.jpeg'))
        #imgcat(result1, height=10)
        #imgcat(result2, height=10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkout', default='../outputs/weights/cub_multi_cam',
            type=str)
    parser.add_argument('--data_root', default='/data/CUB_200_2011', type=str)
    parser.add_argument('--embedding_size', default=256, type=int)
    args = parser.parse_args()
    
    checkout_net = os.path.join(args.checkout, 'best_net.pth')
    checkout_loss1 = os.path.join(args.checkout, 'best_loss1.pth')
    checkout_loss2 = os.path.join(args.checkout, 'best_loss2.pth')
    
    net = MultiNet()
    net.load_state_dict(torch.load(checkout_net))
    net.cuda()
    net.eval()

    loss_fn1 = ProxyAnchorLoss(num_classes=100,
            embedding_size=args.embedding_size)
    loss_fn1.load_state_dict(torch.load(checkout_loss1))
    loss_fn1.cuda()


    loss_fn2 = ProxyAnchorLoss(num_classes=100,
            embedding_size=args.embedding_size)
    loss_fn2.load_state_dict(torch.load(checkout_loss2))
    loss_fn2.cuda()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
 
    ds = CUB2011(root=args.data_root, train=True)
    ds.transform = transform
    dl = DataLoader(dataset=ds, batch_size=256, shuffle=True, num_workers=4,
            pin_memory=True)

    cam(net, loss_fn1, loss_fn2, dl)


