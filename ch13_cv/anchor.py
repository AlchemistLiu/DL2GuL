# 锚框
import torch
from torch._C import device
import torchvision
from d2l import torch as d2l

'''
想钓鱼一样，广撒网(生成一堆锚框),当锚框碰到真实的边界框时，保留，删除其他锚框
锚框就相当于分类检测中可能存在的分类，真实的边界框就是分类检测中的目标
预测每个锚框成为真实边界框的概率
'''
# 打印精度
torch.set_printoptions(2)

# 在输入图像中生成不同的锚框
# 全生成太离谱了
# 先固定锚框的宽，试试不同宽高比，再固定宽高比，试试不同宽
# 选出一个靠谱点的数据
# (s1,r1),(s1,r2),…,(s1,rm),(s2,r1),(s3,r1),…,(sn,r1).
# 输入图像(data) 尺度列表(size) 宽高比(ratios)
def multibox_prior(data, size, ratios):
    # 生成以每个像素为中心具有不同形状的锚框
    # data (batch , channel, height, width)
    in_height, in_width = data.shape[-2:] # 取出最后两个数据
    # 输入的size,ratios都为一组数据
    device, num_size, num_ratios = data.device, len(size), len(ratios)
    # (s1,r1),(s1,r2),…,(s1,rm),(s2,r1),(s3,r1),…,(sn,r1)的数目，即锚框数量
    boxes_pre_pixel = (num_size + num_ratios - 1)
    # 变成tensor好操作
    size_tensor = torch.tensor(size, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    # 将锚点移动到像素的中心，设置偏移量为0.5
    offset_h, offset_w = 0.5, 0.5
    # 缩放，防止图片大小改变后锚框不变
    steps_h = 1.0 / in_height
    steps_w = 1.0 / in_width

    # 生成锚框的所有中心点
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h