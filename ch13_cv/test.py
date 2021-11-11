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

img = d2l.plt.imread('pytorch\img\catdog.jpg')        
print(d2l.plt.imshow(img).axes)                  