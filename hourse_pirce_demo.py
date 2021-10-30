# 加州房价下载傻瓜版
# 只做大自然的搬运工

import hashlib
import os
import tarfile
import zipfile
import requests
import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l 

# 下载数据和缓存数据集
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

'''
用来下载数据集，将数据集缓存在本地目录（默认情况下为../data）中，并返回下载文件的名称。
如果缓存目录中已经存在此数据集文件，并且其sha-1与存储在DATA_HUB中的相匹配，
将使用缓存的文件，以避免重复的下载。
'''
def download(name, cache_dir=os.path.join('..', 'data')):
    '''下载一个DATA_HUB中的文件，返回本地文件名'''
    assert name in DATA_HUB, f"{name}不存在于{DATA_HUB}."
    url, shal_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        shal = hashlib.shal()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                shal.updata(data)
        if shal.hexdigest() == shal_hash:
            return fname
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

'''
两个额外的实用函数：一个是下载并解压缩一个zip或tar文件，
另一个是将使用的所有数据集从DATA_HUB下载到缓存目录中。
'''
def download_extrat(name, folder=None):
    '''下载并解压zip/tar文件'''
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩。'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def dowenload_all():
    '''下载DATA_HUB中的所有文件。'''
    for name in DATA_HUB:
        download(name)

# 访问和读取数据集
# 下载并缓存数据集
DATA_HUB['kaggle_house_train'] = (  #@save
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (  #@save
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

# 使用pandas分别加载包含训练数据和测试数据的两个CSV文件
train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))
# 查看数据格式
# print(train_data.shape)  (1460, 81)
# print(test_data.shape)   (1459, 80)
# 查看前四个和最后两个特征，以及相应标签
# print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
'''
   Id  MSSubClass MSZoning  LotFrontage SaleType SaleCondition  SalePrice
0   1          60       RL         65.0       WD        Normal     208500
1   2          20       RL         80.0       WD        Normal     181500
2   3          60       RL         68.0       WD        Normal     223500
3   4          70       RL         60.0       WD       Abnorml     140000
'''
# 在将数据送到模型中时，将ID从数据集中删除
all_features = pd.concat((train_data.iloc[:, 1: -1], test_data.iloc[:, 1:]))
# print(all_features.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
'''
   MSSubClass MSZoning  LotFrontage  LotArea  YrSold SaleType SaleCondition
0          60       RL         65.0     8450    2008       WD        Normal
1          20       RL         80.0     9600    2007       WD        Normal
2          60       RL         68.0    11250    2008       WD        Normal
3          70       RL         60.0     9550    2006       WD       Abnorml
'''

# 数据预处理
# 尽量缩小较大的数值
train_data.iloc[:, [4,-1]] = np.log(train_data.iloc[:, [4]] + 1)
# 将所有缺失的值替换为相应特征的平均值。
# 将特征重新缩放到零均值和单位方差来标准化数据。
# 选择所有内容为数值的标签
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
'''
Index(['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
       'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
       'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
       'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
       'MoSold', 'YrSold'],
      dtype='object')
'''
# 对数据进行上文所说的预处理
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 将缺失的数据设为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 用one-hot替代离散值
# `Dummy_na=True` 将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
all_features = pd.get_dummies(all_features, dummy_na=True)
# all_features.shape    (2919, 331)---->(1460+1459, 79-->331)

# 从pandas格式中提取NumPy格式，并将其转换为张量表示
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=d2l.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=d2l.float32)
# 房价lables
train_lables = torch.tensor(
    train_data.SalePrie.values.reshape(-1, 1), dtype=d2l.float32)

# 训练模型（2隐藏层感知机Relu激活函数）
loss = nn.MSELoss()
in_features = train_features.shape[1] #331
num_hidden1, num_hidden2, num_outputs = 150, 75, 1
dropout_1, dropout_2 = 0.5, 0.5

def get_net():
    net = nn.Sequential(nn.Linear(in_features, num_hidden1), nn.ReLU(),
                        nn.Dropout(dropout_1),
                        nn.Linear(num_hidden1, num_hidden2), nn.ReLU(),
                        nn.Dropout(dropout_2),
                        nn.Linear(num_hidden2, num_outputs))
    return net