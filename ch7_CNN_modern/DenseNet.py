# 稠密连接网络
# 稠密块
import torch
from torch import nn
from d2l import torch as d2l

# 定义批量归一化、激活和卷积块
def conv_blok(input_channels, num_channels):
    # BN + 激活 + 3*3卷积
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))

# 一个稠密块由多个卷积块组成，每个卷积块使用相同数量的输出信道。 
# 然而，在前向传播中，我们将每个卷积块的输入和输出在通道维上连结。

class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            # 输出通道*i+输入通道
            layer.append(conv_blok(num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 连接通道维度上每个块的输入和输出
            X = torch.cat((X, Y), dim=1)
        return X

'''
类似res_net,在进入每个dense块时同时输入X从另一分支直接流向下一块的输入，在那里做前向传播的时候
这个输入X与dense块的输出Y在通道维cat一下，变成一个(1, X+Y, h, w)的输入进入下一个块
所以dense网络中的通道变化可以写为：设一个dense块的输入为x通道，其中最后一块conv的输出为y通道,每个conv块输出是y不变的
x->[y--(x与y进行cat)]->x+y->[x+y--(x+y与y进行cat)]->x+2y
'''
blk = DenseBlock(2, 3, 8)
X = torch.rand((1, 3, 16, 16),dtype=torch.float32)
Y = blk(X)
print(Y.shape)
# torch.Size([1, 19, 4, 4])
# (1,3, x, x)->(1, 3+8, x, x)->(1, 11+8, x, x)
# wuhu~
# 卷积块的通道数控制了输出通道数相对于输入通道数的增长，因此也被称为增长率（growth rate）。

# 过度层
# 每个稠密块都会带来通道数的增加，使用过多则会过于复杂化模型
# 过渡层可以用来控制模型复杂度
# 通过1×1卷积层来减小通道数，并使用步幅为 2 的平均汇聚层减半高和宽
def transition_block(input_channels, num_channels):
    return nn.Sequential(
                nn.BatchNorm2d(input_channels), nn.ReLU(),
                nn.Conv2d(input_channels, num_channels, kernel_size=1),
                nn.AvgPool2d(kernel_size=2, stride=2))

blk = transition_block(19, 10)
print(blk(Y).shape)
# torch.Size([1, 19, 4, 4])-->torch.Size([1, 10, 2, 2])

# densenet
# 首先使用同 ResNet 一样的单卷积层和最大汇聚层。
b1 = nn.Sequential( nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                    nn.BatchNorm2d(64), nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# `num_channels`为当前的通道数
num_channels, growth_rate = 64, 32
# 16个denseblock
num_convs_in_dense_blocks = [4, 4, 4, 4]
blks = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    # 上一个稠密块的输出通道数
    num_channels += num_convs * growth_rate
    # 在稠密块之间添加一个转换层，使通道数量减半
    # 判断是不是最后一块
    if i != len(num_convs_in_dense_blocks) - 1:
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2

net = nn.Sequential(
    b1, *blks,
    nn.BatchNorm2d(num_channels), nn.ReLU(),
    nn.AdaptiveMaxPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(num_channels, 10))