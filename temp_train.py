import os
import argparse
import logging

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.backends.cudnn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from lib.datasets import CUB2011
from lib.models import MultiNet
from pytorch_metric_learning.losses import MarginLoss, ProxyAnchorLoss
from pytorch_metric_learning.utils import distributed as pml_dist
from pytorch_metric_learning.samplers import MPerClassSampler
from pytorch_metric_learning.distances import CosineSimilarity
from catalyst.data.sampler import DistributedSamplerWrapper


from torch.autograd import grad
import torch.nn.functional as F
from lib.blocks.norm import L2Norm
from lib.utils.eval_func import eval
from lib.blocks.losses import AuxClsLoss


def train(gpu, args):
    rank = args.nr * args.gpus + gpu

    logger = None
    writer = None
    if rank == 0:
        logging.basicConfig(filename=args.log_root, level=logging.DEBUG)
        logging.getLogger("").addHandler(logging.StreamHandler())
        logger = logging.getLogger(__name__)
        writer = SummaryWriter(args.writer_root)

    process_group = dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(2345)
    # network
    net = MultiNet(root=args.backbone_root).cuda()
    net = DDP(net, device_ids=[gpu], broadcast_buffers=False)
    net.train()

    # loss function
    loss_fn_1 = ProxyAnchorLoss(num_classes=100, embedding_size=768).cuda()
    loss_fn_1 = pml_dist.DistributedLossWrapper(loss=loss_fn_1, device_ids=[gpu])
    loss_fn_2 = ProxyAnchorLoss(num_classes=100, embedding_size=768).cuda()
    loss_fn2 = pml_dist.DistributedLossWrapper(loss=loss_fn_2, device_ids=[gpu])
    loss_fn_aux = AuxClsLoss(cls=100, label_smooting=0.2,
            temperature_scaling=0.5).cuda()

    # optimizer
    optimizer_net = torch.optim.AdamW(net.parameters(), lr=args.lr,
            weight_decay=1e-4)
    scheduler_net = torch.optim.lr_scheduler.StepLR(optimizer_net,
            step_size=5, gamma=0.5)

    loss_params = list(loss_fn_1.parameters()) + list(loss_fn_2.parameters())
    optimizer_loss = torch.optim.AdamW(loss_params, lr=args.lr * 100,
            weight_decay=1e-4)
    scheduler_loss = torch.optim.lr_scheduler.StepLR(optimizer_loss,
            step_size=5, gamma=0.5)

    # data loader
    train_ds = CUB2011(root=args.data_root, train=True)
    balance_sampler = MPerClassSampler(labels=train_ds.cls_list, m=5,
            length_before_new_iter=10000)
    sampler = DistributedSamplerWrapper(sampler=balance_sampler,
            num_replicas=args.world_size, rank=rank)
    train_dl = DataLoader(dataset=train_ds, batch_size=args.b, shuffle=False, num_workers=4, pin_memory=True, sampler=sampler)

    test_ds = CUB2011(root=args.data_root, train=False)
    test_dl = DataLoader(dataset=test_ds, batch_size=512, shuffle=False,
            num_workers=4, pin_memory=True)

    best_recalls = 0.0
    best_e = -1
    for e in range(args.e):
        net.train()

        for m in net.module.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        sampler.set_epoch(e)
        for i, (img, labels) in enumerate(train_dl):
            img = img.cuda()
            labels = labels.cuda()
            b = img.shape[0]

            # train with proxy anchor loss
            # 1. loss calc of proxy anchor
            mac_x, gem_x, cls_out = net.module.forward_proxy(x=img)
            loss_ranking_1 = loss_fn_1(mac_x, labels)
            loss_ranking_2 = loss_fn_2(gem_x, labels)
            loss_ranking = loss_ranking_1 + loss_ranking_2
            loss_aux = loss_fn_aux(cls_out, labels)

            loss = loss_ranking + loss_aux
            optimizer_net.zero_grad()
            optimizer_loss.zero_grad()
            loss.backward()
            optimizer_net.step()
            optimizer_loss.step()

            # 2. cam loss
            mac_input, mac_x, gem_input, gem_x= net.module.forward_cam(x=img)
            mac_x = loss_fn_1(mac_x, labels)
            gem_x = loss_fn_2(gem_x, labels)

            b, c, h, w = mac_input.shape

            alpha_mac = grad(outputs=mac_x, inputs=mac_input,
                    retain_graph=True, create_graph=True)[0]
            alpha_mac = F.relu(alpha_mac).sum(dim=(2, 3))
            cam_mac = alpha_mac[:, :, None, None] * mac_input
            cam_mac = F.relu(cam_mac.sum(dim=1)) / (h * w)

            alpha_gem = grad(outputs=gem_x, inputs=gem_input,
                    retain_graph=True, create_graph=True)[0]
            alpha_gem = F.relu(alpha_gem).sum(dim=(2, 3))
            cam_gem = alpha_gem[:, :, None, None] * gem_input
            cam_gem = F.relu(cam_gem.sum(dim=1)) / (h * w)

            with torch.no_grad():
                w = 100.0
                sigma = cam_ma.view(b, -1).max(dim=1)[0]
                sigma = sigma.view(b, 1, 1)
                mask = 1.0 / (1.0 + torch.exp(-w * (cam_ma - sigma * 0.55)))
                mask[mask < 0.5] = 0.0
                mask[mask >= 0.5] = 1.0
                
            loss_cam = 2 * (torch.minimum(cam_ma, cam_gem) * mask).sum(dim=(1,2)) / \
                    (cam_m + cam_gem).sum(dim=(1, 2))
            loss_cam = 10 * loss_cam.mean()

            optimizer_loss.zero_grad()
            optimizer_net.zero_grad()
            loss_cam.backward()
            optimizer_net.step()

            if writer is not None:
                writer.add_scalar('loss/loss_ranking', loss_ranking.item(), e * len(train_dl) + i)
                writer.add_scalar('loss/loss_ranking_1', loss_ranking_1.item(), e * len(train_dl) + i)
                writer.add_scalar('loss/loss_ranking_2', loss_ranking_2.item(), e * len(train_dl) + i)
                writer.add_scalar('loss/loss_aux', loss_aux.item(), e * len(train_dl) + i)
                writer.add_scalar('loss/loss_cam', loss_cam.item(), e * len(train_dl) + i)
                logger.info(f'[{e}/{args.e}][{i}/{len(train_dl)}] loss_ranking: {loss_ranking.item()} '
                        f'loss_aux: {loss_aux.item()}')
        if rank == 0:
            torch.save(net.module.state_dict(),
                    f'../outputs/weights/desc-{args.num_desc}_epoch-{e}_net.pth')
            torch.save(loss_fn_1.loss.module.state_dict(), 
                    f'../outputs/weights/desc-{args.num_desc}_epoch-{e}_loss_1.pth')
            torch.save(loss_fn_1.loss.module.state_dict(), 
                    f'../outputs/weights/desc-{args.num_desc}_epoch-{e}_loss_2.pth')
            logger.info('=============== save ===============')
        scheduler_net.step()
        scheduler_loss.step()

        net.eval()
        recalls = eval(net.module, test_dl)

        if writer is not None:
            writer.add_scalar('eval/recall', recalls[0], e)

        if recalls[0] > best_recalls:
            best_recalls = recalls[0]
            best_e = e

    print(best_recalls)
    print(best_e)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vgpus', default='0', type=str)
    parser.add_argument('--nodes', default=1, type=int)
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--nr', default=0, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--b', default=100, type=int, help='batch size')
    parser.add_argument('--e', default=10, type=int, help='epoch')
    parser.add_argument('--data_root', default='/data/CUB_200_2011', type=str)
    parser.add_argument('--backbone_root', default='../outputs/pretrained/resnet50.pth', type=str)
    parser.add_argument('--log_root', default='../outputs/log/temp.log', type=str)
    parser.add_argument('--writer_root', default='../outputs/tensorboard/temp', type=str)
    parser.add_argument('--num_desc', default=5, type=int)
    args = parser.parse_args()

    args.world_size = args.nodes * args.gpus
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.vgpus
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    mp.spawn(train, nprocs=args.gpus, args=(args, ))


if __name__ == '__main__':
    #os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2'
    main()
