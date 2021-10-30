import torch
from torch import nn
from d2l import torch as d2l

dropout_1, dropout_2 = 0.2, 0.5
# 实例化一个多层感知机

net = nn.Sequential( # 隐藏层1 
                    nn.Linear(8, 16), nn.ReLU(),
                    # 第一个dropout层
                    # nn.Dropout(dropout_1),
                    # 隐藏层2
                    nn.Linear(16, 8), nn.ReLU(),
                    # 第二个dropout层
                    # nn.Dropout(dropout_2),
                    nn.Linear(8, 4))
# def init_weights(m):
#     if type(m) == nn.Linear:
#         return 0
# net.apply(init_weights)
# 访问参数
p = [net[0].weight.data for i in range(len(net))]
print(p)  
