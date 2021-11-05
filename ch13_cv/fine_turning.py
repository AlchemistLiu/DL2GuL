# 微调(数据迁移)
import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.transforms.transforms import RandomHorizontalFlip

# 热狗识别
d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip',
                          'fba480ffa8aa7e0febbb511d181409f899b9baa5')

data_dir = d2l.download_extract('hotdog')

train_img = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_img = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))

# 看一眼
hotdogs = [train_img[i][0] for i in range(8)]
not_hotdogs = [train_img[-i - 1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)
plt.show()

# class torchvision.transforms.Normalize(mean, std)
# 给定均值：(R,G,B) 方差z（R，G，B），将会把Tensor正则化
# image-net上就这么搞了
normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])

# 数据增广
train_augs = torchvision.transforms.Compose([
    # 大小匹配
    torchvision.transforms.RandomResizedCrop(224),
    # 随机水平翻转
    torchvision.transforms.RandomHorizontalFlip(),
    transforms.transforms.ToTensor(), normalize])

test_augs = torchvision.transforms.Compose([
    # 大小匹配(短边弄成256，长边跟着变)
    torchvision.transforms.Resize(256),
    # 移到中心
    torchvision.transforms.CenterCrop(224),
    transforms.transforms.ToTensor(), normalize])

# 定义+初始化模型
# pretrained=True 拿来模型的同时也拿来params
pretrained_net = torchvision.models.resnet18(pretrained=True)
# 查看最后一层(分类器)的情况
print(pretrained_net.fc)

# 开始微调
finetune_net = torchvision.models.resnet18(pretrained_net=True)
# 修改分类器输出为两类(hotdogs/not_hotdogs)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
# 初始化最后一层的params
nn.init.xavier_uniform_(finetune_net.fc.weight)

def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5,
                      param_group=True):
    train_iter = torch.utils.data.Dataloder(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'), transform=test_augs),
        batch_size=batch_size)
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction='none')
    if param_group:
        # 拿出除最后一层(分类器)的所有层
        params_1x = [param for name, param in net.named_parameters()
             if name not in ["fc.weight", "fc.bias"]]
        trainer = torch.optim.SGD([{'params': params_1x},
                                   {'params': net.fc.parameters(),
                                   # 希望最后一层学更快
                                    'lr': learning_rate * 10}],
                                lr=learning_rate, weight_decay=0.001)
        '''class torch.optim.Optimizer(params, defaults) [source]
           params (iterable) —— Variable 或者 dict的iterable。指定了什么参数应当被优化。
           defaults —— (dict)：包含了优化选项默认值的字典（一个参数组没有指定的参数选项将会使用默认值）。'''
    # 正常分支
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                  weight_decay=0.001)
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,devices=d2l.try_all_gpus())

train_fine_tuning(finetune_net, 5e-5)