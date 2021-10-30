# AlxNet
import torch
from torch import nn
from d2l import torch as d2l

dropout_1 = 0.5
dropout_2 = 0.5
# 用fashion_mnist，输入通道为1
net = nn.Sequential(
    # 使用一个11*11的更大窗口来捕捉对象。
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2), 
    # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(), 
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 继续增加通道
    # padding=1为了不让输出大小改变    x-3+2+1 = x
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # 全连接使用dropout
    nn.Linear(1*256*5*5, 4096), nn.ReLU(),
    nn.Dropout(dropout_1),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(dropout_2),
    # fashion_mnist一共就十类
    nn.Linear(4096, 10))
X = torch.randn((1, 1, 224, 224), dtype=torch.float32)
# X = torch.randn(1, 1, 224, 224)
for lay in net:
    X = lay(X)
    print(lay.__class__.__name__, 'outpute shape:\t', X.shape)

'''
Conv2d outpute shape:    torch.Size([1, 96, 54, 54])
ReLU outpute shape:      torch.Size([1, 96, 54, 54])
MaxPool2d outpute shape: torch.Size([1, 96, 26, 26])
Conv2d outpute shape:    torch.Size([1, 256, 26, 26])
ReLU outpute shape:      torch.Size([1, 256, 26, 26])
MaxPool2d outpute shape: torch.Size([1, 256, 12, 12])
Conv2d outpute shape:    torch.Size([1, 384, 12, 12])
ReLU outpute shape:      torch.Size([1, 384, 12, 12])
Conv2d outpute shape:    torch.Size([1, 384, 12, 12])
ReLU outpute shape:      torch.Size([1, 384, 12, 12])
Conv2d outpute shape:    torch.Size([1, 256, 12, 12])
ReLU outpute shape:      torch.Size([1, 256, 12, 12])
MaxPool2d outpute shape: torch.Size([1, 256, 5, 5])
Flatten outpute shape:   torch.Size([1, 6400])
Linear outpute shape:    torch.Size([1, 4096])
ReLU outpute shape:      torch.Size([1, 4096])
Dropout outpute shape:   torch.Size([1, 4096])
Linear outpute shape:    torch.Size([1, 4096])
ReLU outpute shape:      torch.Size([1, 4096])
Dropout outpute shape:   torch.Size([1, 4096])
Linear outpute shape:    torch.Size([1, 10])
'''

batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
# 
lr, num_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu(1))
torch.save(net.state_dict(), 'AlxNet.params')
# loss 0.331, train acc 0.879, test acc 0.882