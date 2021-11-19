# 转置卷积

import torch
from torch import nn
from d2l import torch as d2l
from torch.nn.modules import padding

# 乞丐版 无padding stride = 1
def trans_conv(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # 按元素乘法
            Y[i:i + h, j:j + w] += X[i, j] * K
    return Y

X = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
# print(trans_conv(X, K))
# tensor([[ 0.,  0.,  1.], 
#         [ 0.,  4.,  6.], 
#         [ 4., 12.,  9.]])

# 调包侠！
X, K = X.reshape(1, 1, 2, 2), X.reshape(1, 1, 2, 2)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
tconv.weight.data = K
# print(tconv(X))
# tensor([[[[ 0.,  0.,  1.],
#           [ 0.,  4.,  6.],
#           [ 4., 12.,  9.]]]], grad_fn=<SlowConvTranspose2DBackward>)

# padding & stride

tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, padding=1, bias=False)
tconv.weight.data = K
# print(tconv(X))
# padding=1 把最外面一圈删掉
# tensor([[[[4.]]]], grad_fn=<SlowConvTranspose2DBackward>)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)
tconv.weight.data = K
# print(tconv(X))
# tensor([[[[0., 0., 0., 1.],
#           [0., 0., 2., 3.],
#           [0., 2., 0., 3.],
#           [4., 6., 6., 9.]]]], grad_fn=<SlowConvTranspose2DBackward>)

# 多通道
X = torch.rand(size=(1, 10, 16, 16))
conv = nn.Conv2d(10, 20, kernel_size=5, padding=2, stride=3)
tconv = nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2, stride=3)
# print(tconv(conv(X)).shape == X.shape)
# True
