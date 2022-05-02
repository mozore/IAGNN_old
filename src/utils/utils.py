import pickle

import numpy as np
import torch


# 划分验证集
def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set  # (item seq, lab)
    n_samples = len(train_set_x)  # 总长度
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


# 读取数据集
def read_dataset(dataset_dir):
    train_set = pickle.load(open(dataset_dir + '/train.txt', 'rb'))
    test_set = pickle.load(open(dataset_dir + '/test.txt', 'rb'))
    num_items = pickle.load(open(dataset_dir + '/num_items.txt', 'rb'))
    return train_set, test_set, num_items


# 找出最长的序列，填充其他为 item_tail
def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]  # 计算所有序列的长度
    max_len = max(us_lens)  # 取最大的len
    us_pois = [upois[::-1] + item_tail * (max_len - le) for upois, le in zip(all_usr_pois, us_lens)]  # [3,4,5,0,0,0]
    us_msks = [[1] * le + [0] * (max_len - le) for le in us_lens]  # [1,1,1,0,0,0]
    return us_pois, us_msks, max_len


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


# the solution comes from https://discuss.pytorch.org/t/batched-index-select/9115
def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)
