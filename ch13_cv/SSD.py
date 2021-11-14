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
# torch.Size([2, 64, 32, 32]) 高宽减小八倍

# 完整的模型(五个模块)
# 每个块生成的特征图既用于生成锚框，又用于预测这些锚框的类别和偏移量
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1, 1))
    # blk2, blk3
    else:
        blk = down_sample_blk(128, 128)
    return blk

# 为每个块定义前向计算
# 输出包括：(1)CNN特征图 Y
# (2)在当前尺度下根据Y生成的锚框预测类别
# (3)预测的这些锚框的偏移量(基于Y)
# 预测网络只需要前向传播，在计算损失函数的时候才需要锚框
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    # 这里应该是生成的特征图
    Y = blk(X)
    # 对特征图中的像素生成锚框
    # 这里其实只需要fmapY的高宽和size, ratio 对Y具体的值其实不关心
    # 生成锚框
    anchors = d2l.multibox_prior(Y ,sizes=size, ratios=ratio)
    # 在当前尺度下根据Y生成的锚框预测类别
    cls_preds = cls_predictor(Y)
    # 预测的这些锚框的偏移量
    bbox_preds = bbox_predictor(Y)
    return  (Y, anchors, cls_preds, bbox_preds)

# 超参数
# 因为有五个块(base_net, down_sample_blk*3, AdaptiveMaxPool2d)
# [锚框大小, 这块和下一块锚框大小的均值] (乘起来开根号)
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
# 别问，用就对了
ratios = [[1, 2, 0.5]] * 5
# 每个像素维中心，生成四个锚框，具体为什么看anchor.py中的
# (s1,r1),(s1,r2),…,(s1,rm),(s2,r1),(s3,r1),…,(sn,r1)
num_anchors = len(sizes[0]) + len(ratios[0]) - 1

# f'' 表示字符串内支持大括号内的python表达式
# 完整的模型
class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        # 每个blk输出通道数
        idx_to_in_channels = [64, 128, 128, 128, 128]
        # 5次预测
        for i in range(5):
            # setattr(object, name, value) 用于设置属性值 
            # 如果属性不存在会创建一个新的对象属性，并对属性赋值
            # 对每个stage定义相应的网络，cls_pred, bbox_pred
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', 
                    cls_predictor(idx_to_in_channels[i], num_anchors, num_classes))
            setattr(self, f'bbox_{i}', 
                    bbox_predictor(idx_to_in_channels[i], num_anchors))
    
    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5,  [None] * 5, [None] * 5
        # 每个模块里面走一遍
        for i in range(5):
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i], 
                getattr(self,f'cls_{i}'), getattr(self,f'bbox_{i}'))
        # 处理数据格式
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        # self.num_classes + 1 就是锚框的类别值，拿出来以后softmax方便一点
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        # 最后需要的东西其实无关X
        return anchors, cls_preds, bbox_preds

# 测试前向计算
# 就一个香蕉类别
net = TinySSD(num_classes=1)
X = torch.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape)
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)
'''
5444-->所有阶段生成的anchor数
4-->一个anchor需要四个值来定义(类似(左上, 右下))
output anchors: torch.Size([1, 5444, 4])

2-->输入类别加上背景数
5444-->多少个锚框
output class preds: torch.Size([32, 5444, 2])

21776-->5444*4-->对每个锚框做四个预测(offset)
output bbox preds: torch.Size([32, 21776])
'''

# 读取香蕉数据集
batch_size = 32
train_iter, _ = d2l.load_data_bananas(batch_size)

# 参数初始化并且定义优化算法
device, net = d2l.try_gpu(), TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)

# 损失函数和评价函数
# 分类预测
cls_loss = nn.CrossEntropyLoss(reduction='none')
# 预测值可能相差较远，所以用L1
bbox_loss = nn.L1Loss(reduction='none')

# 
def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    # (-1, num_classes) 把批量大小维和锚框个数维放在一起
    clss = cls_loss(cls_preds.reshape(-1, num_classes), 
                   cls_labels.reshape(-1)).reshape(batch_size, -1)
    # bbox_masks是当锚框对应的是背景框是mask等于0，否则等于1
    # 把背景框拿掉
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1)
    return clss + bbox

def cls_eval(cls_preds, cls_labels):
    return float((cls_preds.argmax(dim=1).type(cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())

# 训练
num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['class error', 'bbox mae'])
net = net.to(device)

for epoch in range(num_epochs):
    metric = d2l.Accumulator(4)
    net.train()
    for features, target in train_iter:
        # 训练精确度的和，训练精确度的和中的示例数
        # 绝对误差的和，绝对误差的和中的示例数
        timer.start()
        trainer.zero_grad()
        X, Y = features.to(device), target.to(device)
        # 生成多尺度的锚框，为每个锚框预测类别和偏移量
        anchors, cls_preds, bbox_preds = net(X)
        bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
        l.mean().backward()
        trainer.step()
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.numel())
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')

# 预测
X = torchvision.io.read_image('pytorch\img\banana.jpg').unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()

def predict(X):
    # 预测模式
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

output = predict(X)

# 筛选所有置信度不低于 0.9 的边界框，做为最终输出
def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

display(img, output.cpu(), threshold=0.9)