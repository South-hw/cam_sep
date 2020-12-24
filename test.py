import os
import argparse
import torch
from torch.utils.data import DataLoader

from lib.models import MultiNet
from lib.utils.eval_func import eval
from lib.datasets import CUB2011


def main(args):
    net = MultiNet(num_desc=args.num_desc)
    net.load_state_dict(torch.load(f'../outputs/weights/desc-{args.num_desc}_epoch-{args.cur_e}_net.pth'))
    net.cuda()
    net.eval()

    ds = CUB2011(root=args.data_root, train=args.train)
    dl = DataLoader(dataset=ds, batch_size=args.b_size, shuffle=False, num_workers=4, pin_memory=True) 
    recalls = eval(net, dl)
    print(recalls)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_e', default=0, type=int)
    parser.add_argument('--end_e', default=25, type=int)
    parser.add_argument('--gpu', default='1', type=str)
    parser.add_argument('--num_desc', default=5, type=int)
    parser.add_argument('--b_size', default=1024, type=int)
    parser.add_argument('--data_root', default='/data/CUB_200_2011', type=str)
    parser.add_argument('--train', action='store_true')

    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    for i in range(args.start_e, args.end_e):
        args.cur_e = i
        main(args)
