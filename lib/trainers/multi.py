import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torch.autograd import grad
from tqdm import tqdm

from lib.trainers import BaseTrainer


class MultiTrainer(BaseTrainer):
    def __init__(self, **kwarge):
        self.net = kwarge['net']

        self.loss_ranking_fn1 = kwarge['loss_ranking_fn1']
        self.loss_ranking_fn2 = kwarge['loss_ranking_fn2']
        self.loss_ranking_fn3 = kwarge['loss_ranking_fn3']
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

    def train(self, freeze=True):
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

            # ranking loss
            out1, out2, layer1, layer2, layer3, layer4 = self.net.module.forward_proxy(imgs)
            out3 = torch.cat((out1, out2), dim=1)
            loss_ranking1 = self.loss_ranking_fn1(out1, labels)
            loss_ranking2 = self.loss_ranking_fn2(out2, labels)
            loss_ranking3 = self.loss_ranking_fn3(out3, labels)

            # cam loss
            loss_cam1 = self.calc_cam_loss(loss1=loss_ranking1,
                    loss2=loss_ranking2, feature=layer1)
            loss_cam2 = self.calc_cam_loss(loss1=loss_ranking1,
                    loss2=loss_ranking2, feature=layer2)
            loss_cam3 = self.calc_cam_loss(loss1=loss_ranking1,
                    loss2=loss_ranking2, feature=layer3)
            loss_cam4 = self.calc_cam_loss(loss1=loss_ranking1,
                    loss2=loss_ranking2, feature=layer4)
            #loss_cam1 = loss_cam2 = loss_cam3 =torch.tensor(0.0).float().cuda()

            loss_ranking = loss_ranking1 + loss_ranking2 + 0.0 * loss_ranking3
            loss_cam = 0.0 * loss_cam1 + 0.0 * loss_cam2 + loss_cam3 + loss_cam4
            loss= loss_ranking + 0.1 * loss_cam

            self.optimizer_loss.zero_grad()
            self.optimizer_net.zero_grad()
            loss.backward()
            self.optimizer_net.step()
            self.optimizer_loss.step()
            
            #loss_cam = torch.tensor(0.0).cuda()
            if self.writer is not None:
                self.writer.add_scalar('loss/loss',
                        loss.item(), self.cur_iter)
                self.writer.add_scalar('loss/loss_ranking1',
                        loss_ranking1.item(), self.cur_iter)
                self.writer.add_scalar('loss/loss_ranking2',
                        loss_ranking2.item(), self.cur_iter)
                self.writer.add_scalar('loss/loss_ranking3',
                        loss_ranking3.item(), self.cur_iter)
                self.writer.add_scalar('loss/loss_cam', loss_cam.item(),
                        self.cur_iter)

                self.writer.add_scalar('loss/loss_cam1', loss_cam1.item(),
                        self.cur_iter)
                self.writer.add_scalar('loss/loss_cam2', loss_cam2.item(),
                        self.cur_iter)
                self.writer.add_scalar('loss/loss_cam3', loss_cam3.item(),
                        self.cur_iter)
                self.writer.add_scalar('loss/loss_cam4', loss_cam4.item(),
                        self.cur_iter)
                self.cur_iter += 1

        if self.writer is not None:
            torch.save(self.net.module.state_dict(),
                    os.path.join(self.save_root, f'cam_net_epoch_{self.cur_e}_.pth'))
            torch.save(self.loss_ranking_fn1.loss.module.state_dict(),
                    os.path.join(self.save_root, f'cam_loss1_epoch_{self.cur_e}_.pth'))
            torch.save(self.loss_ranking_fn2.loss.module.state_dict(),
                    os.path.join(self.save_root, f'cam_loss2_epoch_{self.cur_e}_.pth'))
        self.scheduler_net.step()
        self.scheduler_loss.step()
        self.cur_e += 1

    def calc_cam_loss(self, loss1, loss2, feature):
        b, c, h, w = feature.shape

        alpha1 = grad(outputs=loss1, inputs=feature, retain_graph=True,
                create_graph=True)[0]
        alpha1 = F.relu(alpha1).sum(dim=(2, 3))
        cam1 = alpha1[:, :, None, None] * feature
        cam1 = F.relu(cam1.sum(dim=1)) / (h * w)

        alpha2 = grad(outputs=loss2, inputs=feature, retain_graph=True,
                create_graph=True)[0]
        alpha2 = F.relu(alpha2).sum(dim=(2, 3))
        cam2 = alpha2[:, :, None, None] * feature
        cam2 = F.relu(cam2.sum(dim=1)) / (h * w)

        # mask
        with torch.no_grad():
            w = 100.0
            sigma = cam1.view(b, -1).max(dim=1)[0]
            sigma = sigma.view(b, 1, 1)
            mask = 1.0 / (1.0 + torch.exp(-w * (cam1 - sigma * 0.55)))
            mask[mask < 0.5] = 0.0
            mask[mask >= 0.5] = 1.0

        loss_cam = 2 * (torch.minimum(cam1, cam2) * mask).sum(dim=(1, 2)) / \
            (cam1 + cam2).sum(dim=(1, 2))
        loss_cam = loss_cam.mean()

        return loss_cam




