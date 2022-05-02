import sys
sys.path.insert(0, '../')


import argparse
from models.IAGNN import IAGNN
from utils.train import TrainRunner
from utils.data_loader import SessionSet
from torch.utils.data.dataloader import DataLoader
from utils.utils import read_dataset, trans_to_cuda
import torch
import numpy as np
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample',
                    help='dataset name: diginetica/yoochoose1_64/sample/nowplaying/ddata')
parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=100, help='hidden state size')
parser.add_argument('--epochs', type=int, default=20, help='the number of epochs to train for')
parser.add_argument('--n_intents', type=int, default=40, help='the number of intents')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # 0.001
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # 0.00001
parser.add_argument('--patience', type=int, default=3, help='the number of epoch to wait before early stop ')
parser.add_argument('--alpha', type=float, default=0.1, help='intent graph aggregate self information')
parser.add_argument('--validation', action='store_true', help='validation')
opt = parser.parse_args()


def main():
    setup_seed(123456)
    print(opt)
    dataset_dir = 'datasets/' + opt.dataset
    train_data, test_data, n_items = read_dataset(dataset_dir)

    train_set = SessionSet(train_data)
    test_set = SessionSet(test_data)
    max_len = max(train_set.max_len, test_set.max_len)
    train_loader = DataLoader(train_set, num_workers=4, batch_size=opt.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_set, num_workers=4, batch_size=opt.batch_size, shuffle=False, pin_memory=True)

    model = trans_to_cuda(IAGNN(opt.batch_size, n_items, opt.n_intents, opt.hidden_size, max_len, opt.alpha))

    runner = TrainRunner(
        model,
        train_loader,
        test_loader,
        lr=opt.lr,
        l2=opt.l2,
        lr_dc=opt.lr_dc,
        lr_dc_step=opt.lr_dc_step,
        patience=opt.patience
    )

    runner.train(opt.epochs)


if __name__ == '__main__':
    main()
