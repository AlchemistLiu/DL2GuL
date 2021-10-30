# 从不同层面抽取信息，然后在输出通道维合并
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


# 定义并行块
class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 第一分支 1*1卷积，
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 第二分支，1*1卷积 + 3*3卷积（pad=1）
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 第三分支，1*1卷积 + 5*5卷积（pad=2）
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 第四分支，3*3池化（pad=1） + 1*1卷积
        self.p4_1 = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)
    
    def forward(self, x):
        # 每层中加个relu
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(F.relu(self.p4_1(x))))
        # 将四个分支输出的通道叠加起来
        # 第一维应该是批量大小
        return torch.cat((p1, p2, p3, p4), dim=1)

# GoogleNet
# 第一个模块使用 64 个通道、  7×7  卷积层。
# 输出大小减半
b1 = nn.Sequential( nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),# stride=2宽高减半
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))# 再减半
# Sequential output shape:torch.Size([1, 64, 24, 24])

# 第二个模块使用两个卷积层：第一个卷积层是 64个通道、  1×1  卷积层；
# 第二个卷积层使用将通道数量增加三倍的  3×3  卷积层。 这对应于 Inception 块中的第二条路径。
# 接一个3*3的池化
b2 = nn.Sequential( nn.Conv2d(64, 64, kernel_size=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 192, kernel_size=3, padding=1), # padding=1不改变大小
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))# 这里应该减半
# Sequential output shape:torch.Size([1, 192, 12, 12])

# 第三个模块串联两个完整的Inception块。 
# 第一个 Inception 块的输出通道数为  64+128+32+32=256 ，四个路径之间的输出通道数量比为  64:128:32:32=2:4:1:1 。 
# 第二个和第三个路径首先将输入通道的数量分别减少到  96/192=1/2  和  16/192=1/12 ，然后连接第二个卷积层。
# 第二个 Inception 块的输出通道数增加到  128+192+96+64=480 ，四个路径之间的输出通道数量比为  128:192:96:64=4:6:3:2 。 
# 第二条和第三条路径首先将输入通道的数量分别减少到  128/256=1/2  和  32/256=1/8 。
# 第二分支通道数变化192->96->128
# 第三分支通道数变化192->16->32
# 最后接一个3*3的池化
# in_channels, c1, c2, c3, c4

# Inception块不改变每通道输出大小
b3 = nn.Sequential( Inception(192, 64, (96, 128), (16, 32), 32),
                    Inception(256, 128, (128, 192), (32, 96), 64),
                    nn.MaxPool2d(kernel_size=3, stride=2,padding=1))# 输出大小减半
# Sequential output shape:torch.Size([1, 480, 6, 6])

# 四模块更加复杂， 它串联了5个Inception块，其输出通道数分别是192+208+48+64=512、160+224+64+64=512 、  
# 128+256+64+64=512 、112+288+64+64=528和256+320+128+128=832  。 
# 这些路径的通道数分配和第三模块中的类似，首先是含3×3卷积层的第二条路径输出最多通道，
# 其次是仅含1×1卷积层的第一条路径，之后是含5×5卷积层的第三条路径和含3×3最大汇聚层的第四条路径。 
# 其中第二、第三条路径都会先按比例减小通道数。
# 输出大小，通道测试
b4 = nn.Sequential( Inception(480, 192, (96, 208), (16, 48), 64),
                    Inception(512, 160, (112, 224), (24, 64), 64),
                    Inception(512, 128, (128, 256), (24, 64), 64),
                    Inception(512, 112, (144, 288), (32, 64), 64),
                    Inception(528, 256, (160, 320), (32, 128), 128),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))# 继续减半
# Sequential output shape:         torch.Size([1, 832, 3, 3])

# 第五模块包含输出通道数为256+320+128+128=832和384+384+128+128=1024的两个Inception块。 
# 其中每条路径通道数的分配思路和第三、第四模块中的一致，只是在具体数值上有所不同
# 两个Inception块后紧跟一个全局汇聚层
b5 = nn.Sequential( Inception(832, 256, (160, 320), (32, 128), 128),
                    Inception(832, 384, (192, 384), (48, 128), 128),# Sequential output shape:torch.Size([1, 1024, 3, 3])
                    nn.AdaptiveAvgPool2d((1,1)), # 合并通道Sequential output shape:torch.Size([1, 1024, 1, 1])
                    nn.Flatten())
# Sequential output shape:torch.Size([1, 1280])

net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))

X = torch.rand(size=(1, 1, 96, 96))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)

'''
Sequential output shape:torch.Size([1, 64, 24, 24])
Sequential output shape:torch.Size([1, 192, 12, 12])
Sequential output shape:torch.Size([1, 480, 6, 6])
Sequential output shape:torch.Size([1, 832, 3, 3])
Sequential output shape:torch.Size([1, 1280])
Linear output shape:torch.Size([1, 10])
'''

# 使用 Fashion-MNIST 数据集来训练模型。在训练之前，将图片转换为96×96分辨率。
# lr, num_epochs, batch_size = 0.1, 10, 128
# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
# d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
# loss 0.248, train acc 0.906, test acc 0.853

# 卷积核的变小导致网络的参数规模变小，层数较AlxNet变多但是参数大小从187-->23.9MB