import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import PIL.Image as Image


class CUB2011(Dataset):
    def __init__(self, root, train):
        super(CUB2011, self).__init__()
        self.root = root
        self.train = train

        self.img_list, self.cls_list, self.bb_list = self.get_filelist()
        if self.train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


    def get_filelist(self):
        img_list = os.path.join(self.root, 'images.txt')
        img_list = open(img_list, 'r').readlines()
        img_list = [line.rstrip('\n') for line in img_list]
        img_list = [line.split(' ') for line in img_list]
        img_list = [line[1] for line in img_list]

        cls_list = os.path.join(self.root, 'image_class_labels.txt')
        cls_list = open(cls_list, 'r').readlines()
        cls_list = [line.rstrip('\n') for line in cls_list]
        cls_list = [line.split(' ') for line in cls_list]
        cls_list = [int(line[1]) - 1 for line in cls_list]

        bb_list = os.path.join(self.root, 'bounding_boxes.txt')
        bb_list = open(bb_list, 'r').readlines()
        bb_list = [line.rstrip('\n') for line in bb_list]
        bb_list = [line.split(' ') for line in bb_list]
        bb_list = [[int(float(line[1])), int(float(line[2])), int(float(line[3])), int(float(line[4]))]
                   for line in bb_list]

        if self.train == True:
            img_list = img_list[:5864]
            cls_list = cls_list[:5864]
            bb_list = bb_list[:5864]
        else:
            img_list = img_list[5864:]
            cls_list = cls_list[5864:]
            bb_list = bb_list[5864:]

        return img_list, cls_list, bb_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = os.path.join(self.root, 'images', self.img_list[idx])
        img = Image.open(img).convert('RGB')
        img = self.transform(img)
        cls = self.cls_list[idx]

        return img, cls
