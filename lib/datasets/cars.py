import os
import scipy.io
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import PIL.Image as Image


class Cars196(Dataset):
    def __init__(self, root, train):
        super(Cars196, self).__init__()
        self.root = root
        self.train = train

        self.img_list, self.cls_list, self.bb_list = self.get_filelist()
        if self.train:
            self. transform = transforms.Compose([
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
        root = self.root
        img_list = os.path.join(root, 'cars_annos.mat')
        mat_file = scipy.io.loadmat(img_list)
        mat_file = mat_file.get('annotations')[0]

        img_list = [item[0] for item in mat_file]
        cls_list = [int(item[5]) for item in mat_file]
        bb_list = [[int(item[1]), int(item[2]), int(item[3]), int(item[4])] for item in mat_file]

        if self.train:
            img_list = img_list[:8054]
            cls_list = cls_list[:8054]
            bb_list = bb_list[:8054]
        else:
            img_list = img_list[8054:]
            cls_list = cls_list[8054:]
            bb_list = bb_list[8054:]

        return img_list, cls_list, bb_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = os.path.join(self.root, self.img_list[idx][0])
        img = Image.open(img).convert('RGB')
        img = self.transform(img)
        cls = self.cls_list[idx]

        return img, cls
