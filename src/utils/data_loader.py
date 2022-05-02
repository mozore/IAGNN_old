from torch.utils.data.dataset import Dataset

import numpy as np
import torch

from src.utils.utils import data_masks


class SessionSet(Dataset):
    def __init__(self, data):
        self.data = data
        self.inputs = self.data[0]
        self.labels = self.data[1]
        self.inputs, self.masks, self.max_len = data_masks(self.inputs, [0])
    def __getitem__(self, idx):
        # 假设这个batch内的物品最长为6
        # 假设一条序列input是 10,20,30,20,40,0,
        # mask是1,1,1,1,1,0
        # nodes是 0,10,20,30,40
        # items是0,10,20，30,40,0
        # alias_inputs为 1,2,3,2,4,0
        input, label, mask = self.inputs[idx], self.labels[idx], self.masks[idx]
        max_n_nodes = self.max_len
        nodes = np.unique(input)
        items = nodes.tolist() + (max_n_nodes - len(nodes)) * [0]
        u_A = np.zeros((max_n_nodes, max_n_nodes))  # 初始化矩阵
        for i in np.arange(len(input) - 1):
            u = np.where(nodes == input[i])[0][0]  # 加入自连接
            u_A[u][u] = 2
            # 编号为0是填充项，结束
            if input[i + 1] == 0:
                break
            # np.where返回的是一个tuple，(array([0], dtype = int64),)
            v = np.where(nodes == input[i + 1])[0][0]
            u_A[u][v] = 1  # 令为1,这里是个无向图
            u_A[v][u] = 1
            u_A[u][u] = 2
        alias_input = [np.where(nodes == i)[0][0] for i in input]
        return torch.tensor(alias_input), torch.tensor(u_A), torch.tensor(items), torch.tensor(mask), torch.tensor(
            label), torch.tensor(input)

    def __len__(self):
        return len(self.inputs)
