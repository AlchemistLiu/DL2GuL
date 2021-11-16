import numpy
import torch
from torch import nn
import torchvision
from torchvision import transforms
from d2l import torch as d2l
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

# trans = transforms.ToTensor()
# mnist_train = torchvision.datasets.FashionMNIST(
#     root="../data", train=True, transform=trans, download=True)
# mnist_test = torchvision.datasets.FashionMNIST(
#     root="../data", train=False, transform=trans, download=True)

# img = mnist_train[0][0].numpy().squeeze() * 255.0
# img = Image.fromarray(img)
# img = img.convert('L')
# img.save(r'myDLstudy\DLmeet\test.jpg')
# a = np.array([[1, 2, 3], 
#               [4, 5, 6]])
# b = np.array([[2, 3, 4], [5, 6, 7]])
# print(a[:, 2])
