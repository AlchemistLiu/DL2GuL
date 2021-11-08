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
a = torch.tensor((0.72, 0.5, 0.25))
b = torch.tensor((1, 2, 0.5))
print(a)
print(b)
print(torch.cat((a, b[1:])))