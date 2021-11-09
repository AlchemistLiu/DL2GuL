# 锚框
import re
import torch
import torchvision
from d2l import torch as d2l
from matplotlib import pyplot as plt

# 左上(x, y)和右下(x, y)
# 左上(x, y)和宽高w,h
d2l.set_figsize()
img = d2l.plt.imread('pytorch\img\catdog.jpg')
d2l.plt.imshow(img)
# plt.show()

# 两种表示转换函数
# 是批量处理的
def box_corner_to_center(boxes):
    # 左上(x1, y1)和右下(x2, y2)-->中间(cx, cy)，宽高w,h
    # 取出所有图片的相应位置信息，如所图片的x1，结果是个列向量
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3] 
    # 简单数学
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    # 在最后拼起来，和原来形状相同
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes

def box_center_to_corner(boxes):
    # 中间(cx, cy)，宽高w,h-->左上(x1, y1)和右下(x2, y2)
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5*w
    y1 = cy - 0.5*h
    x2 = cx + 0.5*w
    y2 = cy + 0.5*h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes

# 左上右下法
dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]

boxes = torch.tensor((dog_bbox, cat_bbox))
# 验证一些变换函数
print(box_center_to_corner(box_corner_to_center(boxes)) == boxes)
# tensor([[True, True, True, True],
#        [True, True, True, True]])


def bbox_to_rect(bbox, color):
    # 将边界框 (左上x, 左上y, 右下x, 右下y) 格式转换成 matplotlib 格式：
    # ((左上x, 左上y), 宽, 高)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)

        
# 画框
# 左上右下
def bbox_to_rect(bbox, color):
    return d2l.plt.Rectangle(xy=(bbox[0], bbox[1]),  # 左上点
                            width = bbox[2] - bbox[0],
                            height = bbox[3] - bbox[1],
                            fill = False, 
                            edgecolor = color, 
                            linewidth = 2)

fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'))
plt.show()