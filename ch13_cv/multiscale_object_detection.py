# 多尺度目标检测
from matplotlib import pyplot as plt
import torch
from torch import nn
from d2l import torch as d2l

img = d2l.plt.imread('pytorch\img\catdog.jpg')
h, w = img.shape[:2]
# print(h, w)  561 728

# 在特征图 (fmap) 上生成锚框 (anchors)，每个单位（像素）作为锚框的中心
# 生成大小为 s（假设列表 s 的长度为 1）且宽高比（ ratios ）不同的锚框
# fmap_w fmap_h 特征图的宽高
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # 前两个维度上的值不影响输出
    fmap = torch.zeros((1, 10, fmap_h, fmap_w))
    # multibox_prior()
    # Generate anchor boxes with different shapes centered on each pixel
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    # 图片真实的宽高
    bbox_scale = torch.tensor((w, h, w, h))
    # anchors[0]取出(1, 10, fmap_h, fmap_w)的1那张图片
    d2l.show_bboxes(d2l.plt.imshow(img).axes, anchors[0] * bbox_scale)

# fmap_w=4, fmap_h=4 假设特征的宽高为4  s=[0.15]占图片15%的区域
# display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
# plt.show()
# 将特征图的高度和宽度减小一半 使用较大的锚框
# display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
# plt.show()
# display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
# plt.show()

'''特征图高宽越小，可以适当调大锚框的面积，否则会出现锚框相互覆盖面积过大'''





