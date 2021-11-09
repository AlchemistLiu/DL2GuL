# 锚框
import torch
from d2l import torch as d2l
from matplotlib import pyplot as plt

'''
这节所有的锚框坐标都做了归一化
'''
torch.set_printoptions(2)  # 精简打印精度

'''
想钓鱼一样，广撒网(生成一堆锚框),当锚框碰到真实的边界框时，保留，删除其他锚框
锚框就相当于分类检测中可能存在的分类，真实的边界框就是分类检测中的目标
预测每个锚框成为真实边界框的概率
'''
# 打印精度
# torch.set_printoptions(2)

# 在输入图像中生成不同的锚框
# 全生成太离谱了
# 先固定锚框的宽，试试不同宽高比，再固定宽高比，试试不同宽
# 选出一个靠谱点的数据
# (s1,r1),(s1,r2),…,(s1,rm),(s2,r1),(s3,r1),…,(sn,r1).
# 输入图像(data) 尺度比例列表 规模？(size) 宽高比(ratios)
def multibox_prior(data, sizes, ratios):
    # 生成以每个像素为中心具有不同形状的锚框
    # data (batch , channel, height, width) 可能是这样的，但是最后两维坑定是这俩
    in_height, in_width = data.shape[-2:] # 取出最后两个数据
    # 输入的size,ratios都为一组数据
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    # (s1,r1),(s1,r2),…,(s1,rm),(s2,r1),(s3,r1),…,(sn,r1)的数目，即锚框数量
    boxes_pre_pixel = (num_sizes + num_ratios - 1)
    # 变成tensor好操作
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    # 将锚点移动到像素的中心，设置偏移量为0.5
    offset_h, offset_w = 0.5, 0.5
    ''' *******************挺屌的**************************** '''
    # 缩放，防止图片大小改变后锚框不变
    steps_h = 1.0 / in_height
    steps_w = 1.0 / in_width
    ''' ***************************************************** '''

    # 生成锚框的所有中心点
    # 多大的高/宽生成有多少数据的一维张量，里面时每个像素对应的中心点
    # 从0开始生成
    # h不一定等于w
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    # torch.meshgrid(center_h, center_w) 返回的两个张量， 
    # 大小为图片大小(shape.center_h, shape.center_w)，
    # 每个元素保存的是对应像素的中心点
    shift_y, shift_x = torch.meshgrid(center_h, center_w)
    # 拉平 我也不知道为什么
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # 前面算出来的(num_size + num_ratios - 1)个渔网
    # 确定锚框的大小和坐标(xmin, xmax, ymin, ymax)
    # size_tensor [0.75, 0.50, 0.25]
    # ratio_tensor[1.00, 2.00, 0.50]
    # (s1,r1),(s1,r2),…,(s1,rm),(s2,r1),(s3,r1),…,(sn,r1)
    # *****************************************************
    # 一个精巧绝伦的(num_size + num_ratios - 1)个数据的生成
    # 直接看了我一下午 湿了！
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                  sizes[0] * torch.sqrt(ratio_tensor[1:])))\
                  * in_height / in_width # 这里应该是对矩形图片的处理，我也不知道为啥
                   # 注释掉有时候也能跑，先注释掉，等出问题了再说
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                  sizes[0] / torch.sqrt(ratio_tensor[1:])))
    # 获得半高/宽
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
                                        in_height * in_width, 1) / 2
    # 每个中心点都将有“boxes_per_pixel”个锚框，
    # 所以生成含所有锚框中心的网格，重复了“boxes_per_pixel”次
    out_grid = torch.stack([shift_x, shift_y ,shift_x, shift_y], # 这里重复两次是为了加上前面的半高和半宽
                dim=1).repeat_interleave(boxes_pre_pixel, dim=0) # 在第0维复制相应份数
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0) # 增加维度

img = d2l.plt.imread('pytorch\img\catdog.jpg')
h, w = img.shape[:2]
print(h, w)
X = torch.rand(size=(1, 3, h, w))
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
print(Y.shape)

# 5-->(3+3-1)
boxes = Y.reshape(h, w, 5, 4)
# 看250为中心的那个锚框
print(boxes[250, 250, 0, :])

# 画一下
def show_bboxes(axes, bboxs, labels=None, colors=None):
    '''看一眼'''
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj
    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxs):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(bbox.detach().numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))
            
d2l.set_figsize()
bbox_scale = torch.tensor((w, h, w, h))
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, boxes[400, 250, :, :] * bbox_scale,
            ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
             's=0.75, r=0.5'])
plt.show()

# 计算成对的交并比IoU
def box_iou(boxes1, boxes2):
    # 计算交并比
    # boxes[:, 2] 取所有行的第三列 h
    # boxes(x1, y1, x2, y2)
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # `boxes1`, `boxes2`, `areas1`, `areas2`的形状: 
    # `boxes1`：(boxes1的数量, 4),
    # `boxes2`：(boxes2的数量, 4), 
    # `areas1`：(boxes1的数量,), 
    # `areas2`：(boxes2的数量,)
    '''boxes1/boxes2的面积'''
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    #  inter_upperlefts, inter_lowerrights, inters的形状: 
    # 左上右下
    # (boxes1的数量, boxes2的数量, 2)
    # boxes1[:, None, :2]好像增加了一维
    # boxes1应该是随机生成的锚框， boxes2是手动标记的正确锚框
    '''
    boxes1, boxes2
    tensor([[0.00, 0.10, 0.20, 0.30],
            [0.15, 0.20, 0.40, 0.40],
            [0.63, 0.05, 0.88, 0.98],
            [0.66, 0.45, 0.80, 0.80],
            [0.57, 0.30, 0.92, 0.90]]) 
    tensor([[0.10, 0.08, 0.52, 0.92],
            [0.55, 0.20, 0.90, 0.88]])
    boxes1[:, None, :2], boxes2[:, :2]
    tensor([[[0.00, 0.10]],

            [[0.15, 0.20]],

            [[0.63, 0.05]],

            [[0.66, 0.45]],

            [[0.57, 0.30]]]) 
    torch.Size([5, 1, 2])
    tensor([[0.10, 0.08],
            [0.55, 0.20]])
    torch.Size([2, 2])
    '''
    # 取重叠部分的左上角，max()取出总是偏右, 取出的每个位置对应最大/最小的元素
    # None在这里在矩阵中间加了一维，数值为1
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    # clamp(min=0)限制值非负
    # 重叠区域inter_areas
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # `inter_areas` and `union_areas`的形状: (boxes1的数量, boxes2的数量)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:,None] + areas2 - inter_areas
    return inter_areas / union_areas

# 将真实边界框分给锚框
# 把矩阵看成一个表格，表格行表示锚框，列表示真实框，根据其中(锚框, 真实框)坐标中存的IoU值
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    # 将最接近的真实边界框分配给锚框。
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # 位于第i行和第j列的元素 x_ij 是锚框i和真实边界框j的IoU
    # anchors应该是一堆锚框，ground_truth应该是手动标记的真实锚框数量，用广播操作
    # 计算生成的锚框与每个真实框之间的IoU
    jaccard = box_iou(anchors, ground_truth)
    # 对于每个锚框，分配的真实边界框的张量
    # torch.full((形状), 值):用来return一个确定值的tensor
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long, device=device)
    # 根据阈值，决定是否分配真实边界框
    max_ious, indices = torch.max(jaccard, dim=1)
    # torch.nonzero()  找出tensor中非零的元素的索引
    anc_i = torch.nonzero(max_ious >= 0.5).reshape(-1)
    box_j = indices[max_ious >= 0.5]
    anchors_bbox_map[anc_i] = box_j
    # 丢弃的行和列
    col_discard = torch.full((num_anchors, ), -1)
    row_discard = torch.full((num_gt_boxes, ), -1)
    for _ in range(num_gt_boxes):
        # 变换为索引的值
        max_idx = torch.argmax(jaccard)
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map

# 标记类和偏移
def offset_boxes(anchors, assigned_bb, eps=1e-6):
    # 转换锚框的偏移量
    c_anc = d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    return offset