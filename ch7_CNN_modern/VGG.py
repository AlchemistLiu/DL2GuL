# VGG块
# 卷积核为3*3 pad=1时输出形状不变
import torch
from torch import nn
from d2l import torch as d2l

# 构建VGG块z
def vgg_block(num_convs, in_channels, out_channels):
    # 定义一个层空网络用来存接下来的块
    layers = []
    # 要几层循环几次
    for _ in range(num_convs):
        # 卷积层
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        # 激活函数
        layers.append(nn.ReLU())
        # 确保不因为通道数报错
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

# VGG参数
# for (num_convs, out_channels) in conv_arch:
# 一共五个块，八层
# 通道数变化 1-->64-->128-->256-->256-->512-->512-->512-->512
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        # 这句很关键！
        in_channels = out_channels
    
    # 补全全连接
    return nn.Sequential(*conv_blks,# vgg层,进全连接之前展开,后面多大输入忘了，测试一下
                        nn.Flatten(),# Sequential output shape:         torch.Size([1, 512, 7, 7])
                        nn.Linear(out_channels * 7 * 7, 4096),nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(4096, 4096), nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(4096, 10))
                                    
net = vgg(conv_arch)
X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)
'''
Sequential output shape:         torch.Size([1, 64, 112, 112])
Sequential output shape:         torch.Size([1, 128, 56, 56])
Sequential output shape:         torch.Size([1, 256, 28, 28])
Sequential output shape:         torch.Size([1, 512, 14, 14])
Sequential output shape:         torch.Size([1, 512, 7, 7])
Flatten output shape:    torch.Size([1, 25088])
Linear output shape:     torch.Size([1, 4096])
ReLU output shape:       torch.Size([1, 4096])
Dropout output shape:    torch.Size([1, 4096])
Linear output shape:     torch.Size([1, 4096])
ReLU output shape:       torch.Size([1, 4096])
Dropout output shape:    torch.Size([1, 4096])
Linear output shape:     torch.Size([1, 10])
'''