import hashlib
import os
import tarfile
import zipfile
import requests
import pandas as pd
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

# a = torch.ones((2, 3))
# b = torch.ones((4, 3)) * 3
a = torch.tensor([[[0.00, 0.10]],

                  [[0.15, 0.20]],

                  [[0.63, 0.05]],

                  [[0.66, 0.45]],

                  [[0.57, 0.30]]]) 
b = torch.tensor([[0.10, 0.08],
                  [0.55, 0.20]])
# print(a)
# print(b)
# print(torch.cat((a, b[1:])))
c =torch.tensor([[0.00, 0.10, 0.20, 0.30],
                [0.15, 0.20, 0.40, 0.40],
                [0.63, 0.05, 0.88, 0.98],
                [0.66, 0.45, 0.80, 0.80],
                [0.57, 0.30, 0.92, 0.90]]) 
print(c.shape)
print(a.shape, b.shape)
print(torch.max(a,b))
print(torch.min(a,b))
'''
tensor([[[0.1000, 0.1000],
         [0.5500, 0.2000]],

        [[0.1500, 0.2000],
         [0.5500, 0.2000]],

        [[0.6300, 0.0800],
         [0.6300, 0.2000]],

        [[0.6600, 0.4500],
         [0.6600, 0.4500]],

        [[0.5700, 0.3000],
         [0.5700, 0.3000]]])
tensor([[[0.0000, 0.0800],
         [0.0000, 0.1000]],

        [[0.1000, 0.0800],
         [0.1500, 0.2000]],

        [[0.1000, 0.0500],
         [0.5500, 0.0500]],

        [[0.1000, 0.0800],
         [0.5500, 0.2000]],

        [[0.1000, 0.0800],
         [0.5500, 0.2000]]])
'''