import torch
from torch import nn
from d2l import torch as d2l
from matplotlib import pyplot as plt
import numpy as np
# 网上随便照两张图片裁一裁测试一下
from PIL import Image


class Reshape(torch.nn.Module):
    def forward(self, x):
        # -1表示批量大小不变
        return x.view(-1, 1, 28, 28)

clone_net = torch.nn.Sequential(
    Reshape(),
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
    nn.Linear(120, 84), nn.ReLU(),
    nn.Linear(84, 10))

clone_net.load_state_dict(torch.load(r'myDLstudy\ch6_CNN\NLeNet.params'))
# 输入图片的维度28*28单通道灰度图
# 读取图片
# *****************************************************************
# in_put = torch.tensor(plt.imread(r'myDLstudy\ch6_CNN\T1.png')) 
# print(in_put.shape) # torch.Size([28, 28, 4])
# *****************************************************************
# out_put = clone_net(in_put)
# # out = out_put.numpy()
# print(type(out_put),out_put.shape)

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def cut(p_path, w=28, h=28):
    img = Image.open(p_path)
    resize_img = img.resize((w, h), Image.ANTIALIAS)

    resize_img.save(p_path)
    img = plt.imread(p_path)
    img = rgb2gray(img)
    in_put = torch.tensor(img, dtype=torch.float32)
    plt.imsave(p_path, in_put)
    return in_put/255

in_put = cut(r'myDLstudy\picture\T1.jpeg')
out_put = clone_net(in_put)
print(out_put)

print(out_put.argmax())
'''
text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']


tensor(8)应该是T-shitr
tensor(6)应该是dress
'''