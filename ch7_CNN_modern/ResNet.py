# yyds!
import torch 
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# 定义残差块
class Residual(nn.Module):
    # 分支进来的数据是否需要1x1的卷积层改变通道数
    # num_channels为输出通道数
    # strides用来指定步幅，该变输出图像大小
    def __init__(self, input_channels, num_channels, use_1x1conv=False,
                strides=1):
        super().__init__()
        # kernel_size=3,padding=1不改变大小
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3,
                                stride=strides, padding=1)
        # 这一层只做特征提取，不设置stride
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        # 是否使用1x1卷积
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, 
                                    stride=strides)
        else:
            self.conv3 = None
        # BN层
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        # 可以在这里设定RELU

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        # 合并分支
        Y += X
        return F.relu(Y)
        
'''
# 测试
# 输出高宽不变时
blk = Residual(3,3)
T = torch.rand(4, 3, 6, 6)
Y = blk(T)
print(Y.shape)# torch.Size([4, 3, 6, 6])
# 输出高宽改变时
blk = Residual(3,3,strides=2,use_1x1conv=True)
T = torch.rand(4, 3, 6, 6)
Y = blk(T)
print(Y.shape)# torch.Size([4, 3, 3, 3])
'''


# 在输出通道数为 64、步幅为2的7×7卷积层后，接步幅为2的3×3的最大汇聚层。不同之处在于ResNet每个卷积层后增加了批量归一化层
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
# Sequential output shape:torch.Size([1, 1, 224, 224])

# ResNet_block
# num_residuals残差块数
# 
def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        # 如果是第一块的话需要对高宽做变化，匹配通道数
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

# ResNet使用4个由残差块组成的模块，每个模块使用若干个同样输出通道数的残差块。
# 第一个模块的通道数同输入通道数一致
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))
res_net = nn.Sequential(b1, b2, b3, b4, b5,
                        nn.AdaptiveAvgPool2d((1, 1)), 
                        nn.Flatten(),
                        nn.Linear(512, 10))
# 测试
X = torch.rand(size=(1, 1, 224, 224))
for layer in res_net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)

'''
Sequential output shape:torch.Size([1, 64, 56, 56])
Sequential output shape:torch.Size([1, 64, 56, 56])
Sequential output shape:torch.Size([1, 128, 28, 28])
Sequential output shape:torch.Size([1, 256, 14, 14])
Sequential output shape:torch.Size([1, 512, 7, 7])
AdaptiveAvgPool2d output shape:  torch.Size([1, 512, 1, 1])
Flatten output shape:   torch.Size([1, 512])
Linear output shape:    torch.Size([1, 10])
'''

# 训练
lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(res_net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

torch.save(res_net.state_dict(), 'Res_Net.params')
