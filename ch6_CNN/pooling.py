# 池化层
# (1).可以返回窗口中的最大值或平均值
# (2).作用应该是对卷积输出的结果的位置信息不那么敏感
# eg输入中的某根线向左右平移一下也可以记录下来
# 有填充，步幅，窗口大小
# 不容和通道，输入多少通道输出多少通道
import torch
from torch import nn
from d2l import torch as d2l

# 正向传播
# mode=max/avg     最大/平均
# pool_size二维 (x, y)
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    # 初始化输出    形状计算参考卷积核，暂时不考虑步幅填充
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w +1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i:i+p_h, j:j+p_h].max()
            if mode == 'avg':
                Y[i, j] = X[i:i+p_h, j:j+p_h].mean()
    return Y

X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
# 最大池化
result = pool2d(X, (2, 2))
print(result)
# tensor([[4., 5.],
#         [7., 8.]])
# 平均池化
result = pool2d(X, (2, 2), 'avg')
print(result)
# tensor([[2., 3.],
#         [5., 6.]])

# 填充和步幅
X =torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
# print(X)
# tensor([[[[ 0.,  1.,  2.,  3.],
#           [ 4.,  5.,  6.,  7.],
#           [ 8.,  9., 10., 11.],
#           [12., 13., 14., 15.]]]])
pool2d = nn.MaxPool2d(3)
# stride(int or tuple, optional) - max pooling的窗口移动的步长。默认值是kernel_size
print(pool2d(X))    # tensor([[[[10.]]]])
# 设定填充和步幅
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
# print(pool2d(X))
# tensor([[[[ 5.,  7.],
#           [13., 15.]]]])

# 设定矩形池化窗口，设置特殊的填充和步幅
pool2d = nn.MaxPool2d((2, 3), padding=(1, 1), stride=(2, 3))
# print(pool2d(X))
# tensor([[[[ 1.,  3.],
#           [ 9., 11.],
#           [13., 15.]]]])

# 多通道
X = torch.cat((X, X + 1), 1)
# print(X, X.shape)
# tensor([[[[ 0.,  1.,  2.,  3.],
#           [ 4.,  5.,  6.,  7.],
#           [ 8.,  9., 10., 11.],
#           [12., 13., 14., 15.]],

#          [[ 1.,  2.,  3.,  4.],
#           [ 5.,  6.,  7.,  8.],
#           [ 9., 10., 11., 12.],
#           [13., 14., 15., 16.]]]]) torch.Size([1, 2, 4, 4])
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X), pool2d(X).shape)
# tensor([[[[ 5.,  7.],
#           [13., 15.]],

#          [[ 6.,  8.],
#           [14., 16.]]]]) torch.Size([1, 2, 2, 2])
