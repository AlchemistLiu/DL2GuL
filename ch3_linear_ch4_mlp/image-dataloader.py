from numpy.core.numeric import True_
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

'''
d2l.use_svg_display()

# 下载数据集并读取到内存
trans = transforms.ToTensor()
#读取训练集
mnist_train = torchvision.datasets.FashionMNIST(root="../data",train=True, 
                                                transform=trans, 
                                                download=True)
#读取测试集
mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False,
                                                transform=trans, 
                                                download=True) 
# print(len(mnist_train), len(mnist_test))  60000 10000
# print(mnist_train[0][0].shape)    torch.Size([1, 28, 28])
# 数据集同时包含图片和标签，第一个[0]表示取出图片([1]表示取出标签)

# 在数字标签索引及其文本名称之间进行转换
def get_fashion_mnist_labels(labels):
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# 将样本可视化的函数
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

# X, y = next(iter(data.DataLoader(mnist_train, batch_size=10)))
# show_images(X.reshape(10, 28, 28), 2, 5, titles=get_fashion_mnist_labels(y))

# 读取小批量数据 
batch_size = 256


train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers())

训练时间
timer = d2l.Timer()
扫一遍数据
for X, y in train_iter:
    continue

print(f'{timer.stop():.2f}sec')

'''
def get_dataloader_workers():
    #四进程读取数据
    return 2 

# 整合为一个读取数据函数
# 返回训练集和验证集的数据迭代器
def load_data_fashion_mnist(batch_size, resize = None):
    '''下载数据集'''
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    #下载训练集
    mnist_train = torchvision.datasets.FashionMNIST(root="../data",train=True, 
                                                transform=trans, 
                                                download=True)
    #下载测试集
    mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False,
                                                    transform=trans, 
                                                    download=True) 
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))