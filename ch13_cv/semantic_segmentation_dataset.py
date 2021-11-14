# 读取Pascal VOC2012
import os
import torch
import torchvision
from d2l import torch as d2l
from matplotlib import pyplot as plt

voc_dir = '..\data\VOCdevkit\VOC2012'

def read_voc_images(voc_dir, is_train=True):
    # 读取图像并标注
    # train.txt哪些是训练集
    # val.txt哪些是验证集
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(
            torchvision.io.read_image(os.path.join(voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(
            torchvision.io.read_image(os.path.join(voc_dir, 'SegmentationClass', f'{fname}.png'), mode))
    return features, labels

# train_features, train_labels = read_voc_images(voc_dir, True)

# 看一眼
# n = 5
# imgs = train_features[0:n] + train_labels[0:n]
# imgs = [img.permute(1, 2, 0) for img in imgs]
# d2l.show_images(imgs, 2, n)
# plt.show()

# 列举RGB颜色值和类名
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

#@save
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

# 一个十进制与256进制的转换,避免了使用python的for循环，提高性能
# 查找标签中每个像素的类索引
def voc_colormap2label():
    # 把rgb映射到voc类别
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

def voc_label_indices(colormap, colormap2labels):
    # 将VOC标签中的RGB值映射到它们的类别索引。
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 + colormap[:, :, 2])
    return colormap2labels[idx]

# 栗子
# y = voc_label_indices(train_labels[0], voc_colormap2label())

# print(y[105:115, 130:140])
'''
0是背景，1是飞机
tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]])
'''
# print(VOC_CLASSES[1])
# aeroplane

# 增广
# 使用图像增广中的随机裁剪，裁剪输入图像和标签的相同区域
def voc_rand_crop(feature, label, height, width):
    # 随机裁剪特征和标签图像(剪一个地方)
    # rect 裁剪的bbox
    rect = torchvision.transforms.RandomCrop.get_params(feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label

# imgs = []
# for _ in range(n):
#     imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)

# imgs = [img.permute(1, 2, 0) for img in imgs]
# d2l.show_images(imgs[::2] + imgs[1::2], 2, n)
# plt.show()

# 自定义语义分割数据集
# 通过实现`__getitem__`函数，我们可以任意访问数据集中索引为`idx`的输入图像及其每个像素的类别索引。
# 由于数据集中有些图像的尺寸可能小于随机裁剪所指定的输出尺寸，这些样本可以通过自定义的`filter`函数移除掉。
# 此外，我们还定义了`normalize_image`函数，从而对输入图像的RGB三个通道的值分别做标准化。
class VOCSegDataset(torch.utils.data.Dataset):
    # 用来加载VOC数据集的自定义数据集
    # crop_size-->VOC数据集每张图片大小不一样，那我们就直接把图片截取同样大小的部分出来
    def __init__(self, is_train, crop_size, voc_dir):
        # 给定均值：(R,G,B) 方差：（R，G，B），将会把Tensor正则化
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature) for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read' + str(len(self.features)) + ' examples')
    
    def normalize_image(self, img):
        return self.transform(img.float())

    # 假设图片比crop_size的高宽还要小的话，就扔掉不要了
    def filter(self, imgs):
        return [img for img in imgs if(
            img.shape[1] >= self.crop_size[0] and img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        # 拿到相应的图片，标签，分割
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx], *self.crop_size)
        return (feature, voc_label_indices(label, self.colormap2label))
    
    def __len__(self):
        return len(self.features)

# 读取数据集
crop_size = (320, 480)
voc_train = VOCSegDataset(True, crop_size, voc_dir)
voc_test = VOCSegDataset(False, crop_size, voc_dir)

batch_size = 16
train_iter = torch.utils.data.DataLoader(voc_train, batch_size, shuffle=True,
                                    drop_last=True,
                                    num_workers=d2l.get_dataloader_workers())
for X, Y in train_iter:
    print(X.shape)
    print(Y.shape)
    break

def load_data_voc(batch_size, crop_size):
    """加载VOC语义分割数据集。"""
    # voc_dir = d2l.download_extract('voc2012', os.path.join(
    #     'VOCdevkit', 'VOC2012'))
    voc_dir = '..\data\VOCdevkit\VOC2012'
    num_workers = d2l.get_dataloader_workers()
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, drop_last=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        drop_last=True, num_workers=num_workers)
    return train_iter, test_iter

# crop_size = (320, 480)
# for X, Y in load_data_voc(16, crop_size):
#     print(X.shape)
#     print(Y.shape)
#     break