from PIL import Image
import torch
from torch import nn
from d2l import torch as d2l
from matplotlib import pyplot as plt
import numpy as np

class Reshape(torch.nn.Module):
    def forward(self, x):
        # -1表示批量大小不变
        return x.view(-1, 1, 28, 28)

# 网络
num_inputs, num_hiddens, num_outputs = 28*28, 256, 10
net = nn.Sequential(
    nn.Flatten(), nn.Linear(num_inputs, num_hiddens), nn.ReLU(),
    nn.Linear(num_hiddens, num_outputs))


# 读参数
net.load_state_dict(torch.load(r'myDLstudy\DLmeet\MLP.params'))

# 预测一下
in_put = Image.open(r'myDLstudy\DLmeet\test.jpg')
in_put = np.array(in_put).reshape((1, 1, 28, 28)) / 255
in_put = torch.tensor(in_put).float()
'''
out_put = net(in_put)
preds = d2l.get_fashion_mnist_labels(out_put.argmax(axis=1))
print(preds)
'''


# 取每层参数
# print(net[1].weight.data.shape) torch.Size([256, 784])
# print(net[1].bias.data.shape) torch.Size([256])
l1_weight = net[1].weight.data
l1_bias = net[1].bias.data

def forward(X, weight, bias):
    Y = X * weight 
    return Y + bias

def relu(X):
    mask = X >= 0
    mask = mask.int()
    return X * mask

# print(type(weight), weight.shape)
result = forward(in_put.reshape((1, -1)), l1_weight, l1_bias.reshape((-1, 1)))
result = relu(result)
# print(result.shape) torch.Size([256, 784])

# 拿出来几个（3个）
def get_simple_f(result):
    r_list = []
    strid = int(result.shape[1] / 3) # 261
    # print(result[:, 0:1].shape)
    for i in range(3):
        r_list.append(result[:, i*strid:i*strid+1].squeeze())
    # print(r_list[0].shape)  torch.Size([256])
    return r_list

# 将分类器前的输出汇聚可视化
def get_mean_f(result):
    return result.sum(dim=1) / result.shape[1]

'''
fmap = get_mean_f(result)
fmap = fmap.reshape((16, 16)).numpy()
plt.imshow(fmap)
plt.show()
'''
# print(fmap.shape) torch.Size([256])


'''
fmap = get_simple_f(result)
for i in range(len(fmap)):
    img = fmap[i].reshape((16, 16)).numpy()
    plt.imshow(img)
    plt.show()
'''