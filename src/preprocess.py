import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose/sample')
opt = parser.parse_args()
print(opt)

dataset_dir = 'datasets/' + opt.dataset
filePath = dataset_dir
if opt.dataset == 'sample':
    from utils.preprocess_utils import preprocess_diginetica

    filePath += '/sample_train-item-views.csv'
    preprocess_diginetica(dataset_dir, filePath)
elif opt.dataset == 'diginetica':
    from utils.preprocess_utils import preprocess_diginetica

    filePath += '/train-item-views.csv'
    preprocess_diginetica(dataset_dir, filePath)
elif opt.dataset == 'yoochoose':
    from utils.preprocess_utils import preprocess_yoochoose
    dataset_dir += '1_64'
    filePath += '1_64/yoochoose-clicks.dat'
    preprocess_yoochoose(dataset_dir, filePath)
else:
    print("right")
