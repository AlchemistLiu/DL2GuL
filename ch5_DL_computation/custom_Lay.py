# 构建自定义层
import torch
from torch import nn
import torch.nn.functional as F


# 构造一个一个没有任何参数的自定义层
class CenterLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, X):
        return X - X.mean()

# 验证该层工作正常？
layer = CenterLayer()
print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))
# tensor([-2., -1.,  0.,  1.,  2.]) 与forwar函数设定一致

# 将层作为组件添加到更复杂的模型中
net = nn.Sequential(nn.Linear(8, 128), CenterLayer())
# 验证
Y = net(torch.rand(4, 8))
print(Y.mean()) #tensor(9.3132e-09, grad_fn=<MeanBackward0>)

# 构造一个一个带参数的自定义层
# 包括管理访问，初始化，共享，保存和加载参数模型
class MyLinear(nn.Module):
    # 输入和输出的数量
    def __init__(self, in_units, units):
        super().__init__()
        # 标准正态分布初始化weight
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units, ))  # 这里为什么会有个‘，’,加不加好像都一样、
                                                        #*************************************************#
                                                        #********有','创建的是列向量,没有就是行向量********#
                                                        #*************************************************#
        # print(self.weigh)
        # print(self.bias)
    
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        # 在forward中，使用的F.X函数一般均没有状态参数
        return F.relu(linear)

linear = MyLinear(5, 3)
# print(linear.weight)

# 用自定义层进行正向传播
print(linear(torch.rand(2, 5)))

# 另一种使用方式
net = nn.Sequential(MyLinear(8, 4), MyLinear(4, 2))
print(net(torch.rand(2, 8)))
