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

a = [0, 1, 2, 3, 4, 5, 6]
b = a[-2:]
print(a)
print(b)
print(b.device)