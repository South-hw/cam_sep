import os
import argparse

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.backends.cudnn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from lib.datasets import CUB2011, Cars196
from lib.models import BaseNet, MultiNet
from lib.trainers import BaseTrainer, MultiTrainer
from lib.blocks.losses import AuxClsLoss

from pytorch_metric_learning.losses import ProxyAnchorLoss
from pytorch_metric_learning.utils import distributed as pml_dist
from pytorch_metric_learning.samplers import MPerClassSampler
from catalyst.data.sampler import DistributedSamplerWrapper


def train(gpu, args):
    rank = args.nr * args.gpus + gpu

    writer = None
    if rank == 0:
        writer = SummaryWriter(args.writer_root)

    process_group = dist.init_process_group(backend='nccl',
            init_method='env://', world_size=args.world_size, rank=rank)
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    if args.net =='base':
        net = BaseNet(root=args.backbone_root).cuda()
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net, process_group)
    else:
        net = MultiNet(root=args.backbone_root).cuda()

    net = DDP(net, device_ids=[gpu], output_device=gpu,
            find_unused_parameters=True)
    net.train()

    # loss_function
    if args.net == 'base':
        loss_ranking_fn = ProxyAnchorLoss(num_classes=100,
                embedding_size=args.embedding_size).cuda()
        loss_ranking_fn = pml_dist.DistributedLossWrapper(loss=loss_ranking_fn,
                device_ids=[gpu])
    else:
        loss_ranking_fn1 = ProxyAnchorLoss(num_classes=100,
                embedding_size=args.embedding_size).cuda()
        loss_ranking_fn1 = pml_dist.DistributedLossWrapper(loss=loss_ranking_fn1,
                device_ids=[gpu])
        loss_ranking_fn2 = ProxyAnchorLoss(num_classes=100,
                embedding_size=args.embedding_size).cuda()
        loss_ranking_fn2 = pml_dist.DistributedLossWrapper(loss=loss_ranking_fn2,
                device_ids=[gpu])
        loss_ranking_fn3 = ProxyAnchorLoss(num_classes=100,
                embedding_size=args.embedding_size * 2).cuda()
        loss_ranking_fn3 = pml_dist.DistributedLossWrapper(loss=loss_ranking_fn3,
                device_ids=[gpu])

    # optimizer
    #optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
    #        weight_decay=5e-4)
    optimizer_net = torch.optim.AdamW(net.parameters(), lr=args.lr,
            weight_decay=1e-4)
    scheduler_net = torch.optim.lr_scheduler.StepLR(optimizer_net,
            step_size=5, gamma=0.5)

    if args.net == 'base':
        params = loss_ranking_fn.parameters()
    else:
        params = list(loss_ranking_fn1.parameters()) +\
            list(loss_ranking_fn2.parameters())
    #optimizer_loss = torch.optim.SGD(params, lr=args.lr * 100, momentum=0.9,
    #        weight_decay=5e-4)
    optimizer_loss = torch.optim.AdamW(params, lr=args.lr * 100,
            weight_decay=1e-4)
    scheduler_loss = torch.optim.lr_scheduler.StepLR(optimizer_loss,
            step_size=5, gamma=0.5)

    # loader
    if args.data == 'cub':
        train_ds = CUB2011(root=args.data_root, train=True)
        test_ds = CUB2011(root=args.data_root, train=False)
    elif args.data =='car':
        train_ds = Cars196(root=args.data_root, train=True)
        test_ds = Cars196(root=args.data_root, train=False)
 
    sampler = DistributedSampler(train_ds, num_replicas=args.world_size,
            rank=rank)
    train_dl = DataLoader(dataset=train_ds, batch_size=args.b, shuffle=False, num_workers=4, pin_memory=True, sampler=sampler)
    test_dl = DataLoader(dataset=test_ds, batch_size=512, shuffle=False,
            num_workers=4, pin_memory=True)

    if args.net == 'base':
        trainer = BaseTrainer(net=net, loss_ranking_fn=loss_ranking_fn,
                loss_cls_fn=loss_cls_fn, train_dl=train_dl, test_dl=test_dl,
                optimizer_net=optimizer_net, optimizer_loss=optimizer_loss,
                scheduler_net=scheduler_net, scheduler_loss=scheduler_loss,
                writer=writer, save_root=args.save_root)
    else:
        trainer = MultiTrainer(net=net, loss_ranking_fn1=loss_ranking_fn1,
                loss_ranking_fn2=loss_ranking_fn2, loss_ranking_fn3=loss_ranking_fn3, 
                train_dl=train_dl, test_dl=test_dl, optimizer_net=optimizer_net, optimizer_loss=optimizer_loss,
                scheduler_net=scheduler_net, scheduler_loss=scheduler_loss,
                writer=writer, save_root=args.save_root)

    for e in range(args.e):
        net.train()
        sampler.set_epoch(e)

        if e >= 0:
            trainer.train(freeze=False)
        else:
            trainer.train(freeze=True)
        net.eval()
        trainer.eval()

    print(trainer.best_recalls)
    print(trainer.best_e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='cub', type=str)
    parser.add_argument('--net', default='base', type=str)
    parser.add_argument('--embedding_size', default=512, type=int)
    parser.add_argument('--vgpus', default='0', type=str)
    parser.add_argument('--nodes', default=1, type=int)
    parser.add_argument('--gpus', default=2, type=int)
    parser.add_argument('--nr', default=0, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--b', default=100, type=int, help='batch size')
    parser.add_argument('--e', default=10, type=int, help='epoch')
    parser.add_argument('--data_root', default='/data/CUB_200_2011', type=str)
    parser.add_argument('--backbone_root', default='../outputs/pretrained/resnet50.pth', type=str)
    parser.add_argument('--writer_root', default='../outputs/tensorboard/temp', type=str)
    parser.add_argument('--save_root', default='../outputs/weights/temp')
    args = parser.parse_args()

    args.world_size = args.nodes * args.gpus

    os.environ['CUDA_VISIBLE_DEVICES'] = args.vgpus
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12346'

    mp.spawn(train, nprocs=args.gpus, args=(args, ))


if __name__ == '__main__':
    main()

