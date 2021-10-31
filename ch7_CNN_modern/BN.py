# 批量归一化
# 一般不改变模型精度，但是可以加速模型收敛
# 在微批次上估计每个小批次产生的样本方差?
import torch 
from torch import nn
from d2l import torch as d2l
from matplotlib import pyplot as plt

# BN在干嘛？
# moving_nean,moving_var全局均值/方差？
# eps避免出现0
# momentum用来更新moving_nean,moving_var
def batch_norm(X, gamma, beta, moving_nean, moving_var, eps, momentum):
    # 推理过程中不需要做训练
    if not torch.is_grad_enabled():
        # 像是将预测时输入的数据化为服从n(0, 1) 的正态分布？
        X_hat = (X - moving_nean) / torch.sqrt(moving_var + eps)
    else:
        # 2为全连接层，4为卷积层
        assert len(X.shape) in (2, 4)
        # 全连接情况
        if len(X.shape) == 2:
            # 对小批量样本的输入求平均，结果为(1, n)
            mean = X.mean(dim=0)
            # 简简单单求个方差
            var = ((X - mean) ** 2).mean(dim=0)
        # 卷积情况(2d)
        else:
            # (批量数， 通道数， 高， 宽)
            # (0, 2, 3)沿着通道维度求均值，结果为(1*n*1*1)
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # train时需要做的操作
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 移动加权平均？
        moving_nean = momentum * moving_nean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    # 归一化
    Y = gamma * X_hat + beta
    return Y, moving_nean.data, moving_var.data

# 创建BN层,从零开始
class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super().__init__()
        # 全连接
        if num_dims == 2:
            shape = (1, num_features)# 前面说的结果为(1, n)
        if num_dims == 4:
            shape = (1, num_features, 1, 1)# 前面说的结果为(1*n*1*1)
        self.gamma = nn.Parameter(torch.ones(shape))# gamma不能为0，要不然没法乘
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # 放到同一个设备上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean, self.moving_var, 
            eps=1e-5, momentum=0.9)
        return Y

# LeNet用BN
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4),
    nn.Sigmoid(), nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4),
    nn.Sigmoid(), nn.MaxPool2d(kernel_size=2, stride=2), 
    nn.Flatten(),
    nn.Linear(16* 4 * 4, 120),
    BatchNorm(120, num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, num_dims=2),nn.Sigmoid(),
    nn.Linear(84, 10))

# 训练
lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())


plt.show()
result_gamma = net[1].gamma.reshape((-1, ))
result_beta = net[1].beta.reshape((-1, ))
print(result_gamma)
print(result_beta)
'''
tensor([1.6561, 2.5334, 1.4988, 1.0204, 2.2207, 2.2839],
       grad_fn=<ViewBackward>)
tensor([-0.3777, -2.7621, -0.7913, -0.9835,  1.2398, -0.0079],
       grad_fn=<ViewBackward>)
'''

# 用框架实现
# nn.BatchNorm2d(6), nn.BatchNorm1d(120)
'''
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.MaxPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
    nn.Linear(84, 10))
'''