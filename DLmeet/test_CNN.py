import torch
from torch import nn
from d2l import torch as d2l
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from torch.nn import modules

class Reshape(torch.nn.Module):
    def forward(self, x):
        # -1表示批量大小不变
        return x.view(-1, 1, 28, 28)

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            Reshape(), nn.Conv2d(1, 6, kernel_size=3, padding=1), nn.ReLU(),# 第一层卷积
            nn.AvgPool2d(kernel_size=3, padding=1, stride=2), # 池化1
            nn.Conv2d(6, 12, kernel_size=3, padding=1), nn.ReLU(),# 第二卷积层
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),# 展平
            nn.Linear(588, 10))
    def forward(self, x):
        out = self.layer1(x)
        return out
net = ConvNet()

'''
net = torch.nn.Sequential(
    Reshape(), nn.Conv2d(1, 6, kernel_size=3, padding=1), nn.ReLU(),# 第一层卷积
    nn.AvgPool2d(kernel_size=3, padding=1, stride=2), # 池化1
    nn.Conv2d(6, 12, kernel_size=3, padding=1, stride=2), nn.ReLU(),# 第二卷积层
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),# 展平
    nn.Linear(108, 10))


Reshape output shape:    torch.Size([1, 1, 28, 28])
Conv2d output shape:     torch.Size([1, 6, 28, 28])
ReLU output shape:       torch.Size([1, 6, 28, 28])
AvgPool2d output shape:  torch.Size([1, 6, 14, 14])
Conv2d output shape:     torch.Size([1, 12, 7, 7])
ReLU output shape:       torch.Size([1, 12, 7, 7])
AvgPool2d output shape:  torch.Size([1, 12, 7, 7])
# 拿出这一层
Flatten output shape:    torch.Size([1, 588]) 7*7*12
Linear output shape:     torch.Size([1, 10])
'''
net.load_state_dict(torch.load(r'myDLstudy\DLmeet\CNN.params'))

# 预测一下
in_put = Image.open(r'myDLstudy\DLmeet\test.jpg')
in_put = np.array(in_put).reshape((1, 1, 28, 28)) / 255
in_put = torch.tensor(in_put).float()

out_put = net(in_put)
preds = d2l.get_fashion_mnist_labels(out_put.argmax(axis=1))
print(preds)

# print(net[1].weight.data.shape) torch.Size([6, 1, 3, 3])
# print(net[1].bias.data.shape) torch.Size([6])
# print(c2_weight.shape) torch.Size([12, 6, 3, 3])
# print(c2_bias.shape) torch.Size([12])

'''
print(model)
(0): Reshape()
(1): Conv2d(1, 6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(2): ReLU()
(3): AvgPool2d(kernel_size=3, stride=2, padding=1)
(4): Conv2d(6, 12, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
(5): ReLU()
(6): AvgPool2d(kernel_size=2, stride=2, padding=0)
(7): Flatten(start_dim=1, end_dim=-1)
(8): Linear(in_features=108, out_features=10, bias=True)
'''
out = in_put
j = 0
for i in list(net.layer1):
    out = i(out)
    j += 1
    if j == 7:
        break
out = out.detach().numpy().squeeze()
# print(out.shape) (12, 7, 7)
# print(out.shape[0]) (12)
# a = out[3]
# plt.imshow(a)
# plt.show()
plt.figure()
for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.imshow(out[i])
plt.show()