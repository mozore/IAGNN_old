import pickle
from pathlib import Path

import pandas as pd


# 过滤短的session
def filter_short_sessions(df, min_len=2):
    # sessionId
    #    1       10
    #    2       8
    session_len = df.groupby('sessionId', sort=False).size()
    # print(session_len)
    long_sessions = session_len[session_len >= min_len].index
    df_long = df[df.sessionId.isin(long_sessions)]
    return df_long


# 过滤不常见的item
def filter_infreq_items(df, min_support=5):
    item_support = df.groupby('itemId', sort=False).size()
    freq_items = item_support[item_support >= min_support].index
    df_freq = df[df.itemId.isin(freq_items)]
    return df_freq

def reorder_sessions_by_endtime(df):
    endtime = df.groupby('sessionId', sort=False).timestamp.max()
    df_endtime = endtime.sort_values().reset_index()
    oid2nid = dict(zip(df_endtime.sessionId, df_endtime.index))
    sessionId_new = df.sessionId.map(oid2nid)
    df = df.assign(sessionId=sessionId_new)
    df = df.sort_values(['sessionId', 'timestamp'])
    return df

# 划分数据集 train set 和 test set
def split_by_time(df, timedelta):
    max_time = df.timestamp.max()
    end_time = df.groupby('sessionId').timestamp.max()
    split_time = max_time - timedelta
    train_sids = end_time[end_time < split_time].index
    test_sids = end_time[end_time > split_time].index
    df_train = df[df.sessionId.isin(train_sids)]
    df_test = df[df.sessionId.isin(test_sids)]
    return df_train, df_test


def train_test_split(df, test_split=0.2):
    endtime = df.groupby('sessionId', sort=False).timestamp.max()
    endtime = endtime.sort_values()
    num_tests = int(len(endtime) * test_split)
    test_session_ids = endtime.index[-num_tests:]
    df_train = df[~df.sessionId.isin(test_session_ids)]
    df_test = df[df.sessionId.isin(test_session_ids)]
    return df_train, df_test


def process_augment(sessions, split_num=1):
    out_sessions = []
    labels = []
    for _, session in sessions.items():
        for i in range(1, len(session)):
            labels += [session[-i]]
            out_sessions += [session[:-i]]
    split_num = int(len(out_sessions) / split_num)
    out_sessions = out_sessions[-split_num:]
    labels = labels[-split_num:]
    return out_sessions, labels


def save_sessions(df, filepath, split_num=1):
    df = reorder_sessions_by_endtime(df)
    sessions = df.groupby('sessionId').itemId.apply(list)
    print(len(sessions))
    sessions, labels = process_augment(sessions, split_num)
    print(len(labels))
    pickle.dump((sessions, labels), open(filepath, 'wb'))


# 保存数据集，并过滤掉test中未在train中出现的item
def save_dataset(dataset_dir, df_train, df_test, split_num=1):
    # 过滤test中未在train中出现的item
    # filter items in test but not in train
    df_test = df_test[df_test.itemId.isin(df_train.itemId.unique())]
    df_test = filter_short_sessions(df_test)

    print(f'No. of Clicks: {len(df_train) + len(df_test)}')
    print(f'No. of Items: {df_train.itemId.nunique()}')

    # update itemId
    train_itemId_new, uniques = pd.factorize(df_train.itemId)  # 映射到一个数字，返回一个tuple（数字数组， （unique）原数据的值）
    train_itemId_new += 1  # 从1开始计数
    df_train = df_train.assign(itemId=train_itemId_new)  # 修改itemId值
    oid2nid = {oid: i + 1 for i, oid in enumerate(uniques)}  # 获取id映射的dict，从1开始
    test_itemId_new = df_test.itemId.map(oid2nid)  # 利用dict, 实现series映射
    df_test = df_test.assign(itemId=test_itemId_new)  # 修改df_test

    # save file
    print(f'saving dataset to {dataset_dir}')
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    save_sessions(df_train, dataset_dir / 'train.txt', split_num)
    save_sessions(df_test, dataset_dir / 'test.txt')
    num_items = len(uniques) + 1
    pickle.dump(num_items, open(dataset_dir / 'num_items.txt', 'wb'))


def preprocess_diginetica(dataset_dir, csv_file):
    print(f'reading {csv_file}...')
    df = pd.read_csv(
        csv_file,
        usecols=[0, 2, 3, 4],
        delimiter=';',
        parse_dates=['eventdate'],  # 日期解析
        infer_datetime_format=True,
    )

    print('start preprocessing')
    # 按照session id和 timeframe 排序
    # timeframe (time since the first query in a session, in milliseconds)
    df['timestamp'] = df.eventdate + pd.to_timedelta(df.timeframe, unit='ms')
    df = df.sort_values(['sessionId', 'timestamp'])
    df = filter_short_sessions(df)  # 过滤短的session，默认为2
    df = filter_infreq_items(df)  # 过滤不常见的item
    df = filter_short_sessions(df)  # 再次过滤短session

    df_train, df_test = split_by_time(df, pd.Timedelta(days=7))  # 划分数据集

    save_dataset(dataset_dir, df_train, df_test)  # 保存数据集


def preprocess_yoochoose(dataset_dir, csv_file):
    print(f'reading {csv_file}...')
    df = pd.read_csv(
        csv_file,
        usecols=[0, 1, 2],
        delimiter=',',
        parse_dates=['timestamp'],  # 日期解析
        infer_datetime_format=True,
    )

    print('start preprocessing')
    # 按照session id和 timeframe 排序
    # timeframe (time since the first query in a session, in milliseconds)
    df.rename(columns={'session_id': 'sessionId', 'item_id': 'itemId'}, inplace=True)
    print(df.columns.values)
    df = df.sort_values(['sessionId', 'timestamp'])
    df = filter_short_sessions(df)  # 过滤短的session，默认为2
    df = filter_infreq_items(df)  # 过滤不常见的item
    df = filter_short_sessions(df)  # 再次过滤短session

    df_train, df_test = split_by_time(df, pd.Timedelta(days=1))  # 划分数据集

    save_dataset(dataset_dir, df_train, df_test, 64)  # 保存数据集
