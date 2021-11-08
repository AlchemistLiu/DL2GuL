# 锚框
import torch
from d2l import torch as d2l


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
                  sizes[0] * torch.sqrt(ratio_tensor[1:])))
                   # * in_height / in_width # 这里应该是对矩形图片的处理，我也不知道为啥
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

# img = d2l.plt.imread('pytorch\img\catdog.jpg')
# h, w = img.shape[:2]
# print(h, w)