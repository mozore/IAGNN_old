import numpy as np
import pickle

train = pickle.load(open('yoochoose1_64/train.txt', 'rb'))
test = pickle.load(open('yoochoose1_64/test.txt', 'rb'))

item_dict = {}
print(len(train[0]))


# 统计平均长度
all = 0
for seq in train[0]:
    all += len(seq)
for seq in test[0]:
    all += len(seq)


all += len(train[0]) + len(test[0])
print('avg length: ', all / (len(train[0]) + len(test[0]) * 1.0))

def obtian_sess(train, test):
    train_seq = []
    train_label = []
    test_seq = []
    test_label = []
    item_ctr = 1000001  # item从1开始编号
    # 读取训练session的数据
    seqs = train[0]
    labels = train[1]
    print('start train_sq')
    for seq in seqs:
        outseq = []
        # 就是建立item_dict中 i-> item_ctr的联系
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]
                item_dict[i] = item_ctr
                item_ctr += 1
        outseq = np.subtract(outseq, 1000000).tolist()
        train_seq += [outseq]  # session
    print('start train_label')
    for lab in labels:
        if lab in item_dict:
            l = item_dict[lab]
        else:
            l = item_ctr
            item_dict[lab] = item_ctr
            item_ctr += 1
        train_label += [l]

    seqs = test[0]
    labels = test[1]
    print(item_ctr - 1000000)
    print('start test_sq')
    for seq in seqs:
        outseq = []
        # 就是建立item_dict中 i-> item_ctr的联系
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]
                item_dict[i] = item_ctr
                item_ctr += 1
        outseq = np.subtract(outseq, 1000000).tolist()
        test_seq += [outseq]  # session
    print('start test_label')
    for lab in labels:
        if lab in item_dict:
            l = item_dict[lab]
        else:
            l = item_ctr
            item_dict[lab] = item_ctr
            item_ctr += 1
        test_label += [l]
    # 仅仅采用在训练集出现的物品，实际上从1开始，所以只有（item_ctr-1）个物品
    print(item_ctr - 1000000)  #
    train_label = np.subtract(train_label, 1000000).tolist()
    test_label = np.subtract(test_label, 1000000).tolist()
    return (train_seq, train_label), (test_seq, test_label)

train, test = obtian_sess(train, test)

pickle.dump(train, open('yoochoose1_64/train.txt', 'wb'))
pickle.dump(test, open('yoochoose1_64/test.txt', 'wb'))
