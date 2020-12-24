import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from tqdm import tqdm


class BaseTrainer:
    def __init__(self, **kwarge):
        self.net = kwarge['net']
        self.loss_ranking_fn = kwarge['loss_ranking_fn']
        self.loss_cls_fn = kwarge['loss_cls_fn']

        self.train_dl = kwarge['train_dl']
        self.test_dl = kwarge['test_dl']

        self.optimizer_net = kwarge['optimizer_net']
        self.optimizer_loss = kwarge['optimizer_loss']
        self.scheduler_net = kwarge['scheduler_net']
        self.scheduler_loss = kwarge['scheduler_loss']

        self.writer = kwarge['writer']
        self.save_root = kwarge['save_root'] 

        self.best_recalls = None
        self.best_e = 0
 
        self.cur_iter = 0
        self.cur_e = 0

    def train(self, freeze = True):
        pbar = tqdm(enumerate(self.train_dl), total=len(self.train_dl),
                desc=f'{self.cur_e}')

        self.net.module.freeze_bn()
        if freeze:
            self.net.module.requires_grad_backbone(val=False)
        else:
            self.net.module.requires_grad_backbone(val=True)

        for i, (imgs, labels) in pbar:
            imgs = imgs.cuda()
            labels = labels.cuda()

            out = self.net(imgs)

            loss_ranking = self.loss_ranking_fn(out, labels)
            loss_cls = 0.0

            loss = loss_ranking
            self.optimizer_net.zero_grad()
            self.optimizer_loss.zero_grad()
            loss.backward()
            self.optimizer_net.step()
            self.optimizer_loss.step()

            if self.writer is not None:
                self.writer.add_scalar('loss/loss_ranking',
                        loss_ranking.item(), self.cur_iter)
                self.writer.add_scalar('loss/loss_cls', loss_ranking.item(),
                        self.cur_iter)
                self.writer.add_scalar('loss/total_loss', loss.item(),
                        self.cur_iter)
                self.cur_iter += 1
 
        if self.writer is not None:
            torch.save(self.net.module.state_dict(),
                    os.path.join(self.save_root, f'base_net_epoch_{self.cur_e}_.pth'))
            torch.save(self.loss_ranking_fn.loss.module.state_dict(),
                    os.path.join(self.save_root, f'base_loss_epoch_{self.cur_e}_.pth'))
        self.scheduler_net.step()
        self.scheduler_loss.step()
        self.cur_e += 1

    def eval(self):
        dist, labels = self.calc_dist()
        recalls = self.calc_recalls(dist, labels)

        if self.writer is not None:
            self.writer.add_scalar('eval/recall', recalls[0], self.cur_e)

        if self.best_recalls is None:
            self.best_recalls = recalls
            self.best_e = self.cur_e
        elif recalls[0] > self.best_recalls[0]:
            self.best_recalls = recalls
            self.best_e = self.cur_e

    def calc_dist(self):
        entire_features = []
        entire_labels = []

        with torch.no_grad():
            for i, (imgs, labels) in enumerate(self.test_dl):
                imgs = imgs.cuda()
                labels = labels.cuda()

                features = self.net.module.forward_feature(imgs)
                entire_features.append(features)
                entire_labels.append(labels)

            entire_features = torch.cat(entire_features, dim=0)
            entire_labels = torch.cat(entire_labels, dim=0)

            dist = torch.zeros(size=(entire_features.shape[0],
                entire_features.shape[0])).float().cuda()
            for i in range(entire_features.shape[0]):
                dist[i, :] = F.cosine_similarity(entire_features[i, :].view(1, -1),
                        entire_features, dim=1)

        return dist, entire_labels

    def calc_recalls(self, dist, labels, ranks=(1, 2, 4, 8)):
        recalls = []
        diag = torch.range(start=0, end=dist.shape[0] - 1).long()
        dist[diag, diag] = -1.0

        for k in ranks:
            ind = dist.topk(k)[1]

            mask = (labels.view(-1, 1) == labels[ind]).sum(dim=1)
            correct = (mask > 0).sum()
            r = correct / dist.shape[0]
            recalls.append(r.cpu().item())

        return recalls
