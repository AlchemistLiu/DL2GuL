# 单发多框检测SSD
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# class predictor 预测一个锚框的类别
# num_inputs 输入通道数
# num_anchors 多少个锚框***每一个***像素
# num_classes 多少类
# kernel_size=3, padding=1 大小不变
def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), 
                     kernel_size=3, padding=1)
'''
num_inputs fmap
num_classes + 1 --> 锚框的类别数加1(背景类)
num_anchors 对***每一个***像素生成了多少锚框
num_anchors * (num_classes + 1) 对每一个锚框都要预测 
通道以一个像素为中心的锚框对应的预测值
'''

# 边界框预测层
# 预测与真实bbox的offset
# offset是四个数字 是anchor到真实bbox的偏移
def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)
'''
num_anchors * 4 是每个像素的锚框的offset 扔到每个通道里面
kernel_size=3, padding=1 输入输出大小不变
'''

# 链接多尺度的预测
# blocks 是网络中的一块
def forward(x, blocks):
    return blocks(x)

# ***************************
# fmap的batch_size不变的 是2？
# 这里的2指的是猫和狗两类？
Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
# print(Y1.shape, Y2.shape)
# torch.Size([2, 55, 20, 20]) torch.Size([2, 33, 10, 10])
# 55, 20, 20 对20*20个像素进行55个预测

# 把这些张量转换为更一致的格式
# permute(0, 2, 3, 1)
# 把通道数放到最后(1) 批量大小维不变 高宽相应往高维移位
# start_dim=1 把除批量维 后面高宽和通道展平成一个向量
def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)

# print(concat_preds([Y1, Y2]).shape) torch.Size([2, 25300])

# 高宽减半块
# CNN
def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    # 高宽减半 默认stride=2
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)

# print(forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape)
# torch.Size([2, 10, 10, 10])

# 基本网络块
# 从原始图片抽特征,一直到第一次对fmap做锚框的中间那一节
# 包含三个down_sample_blk
def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i + 1]))
    return nn.Sequential(*blk)

# print(forward(torch.zeros((2, 3, 256, 256)), base_net()).shape)
# torch.Size([2, 64, 32, 32])