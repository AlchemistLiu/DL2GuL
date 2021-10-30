# 神经网络面向对象？的编程方式
# 一块：输入-》正向传播-》输出（保存相应的参数） f(计算梯度)-》反向传播函数（由于自动微分，只需考虑正向）

# 回顾(多层感知机)
import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(),
                    nn.Linear(256, 10))
X = torch.rand(2, 20)
net(X)
# net.__call__(X)
# nn.Sequential()维护由Moudle组成的 |有序| 列表
# net(X)即net.__call__(X)的缩写，用来调用模型来获得模型的输出

# 自定义块
'''
块的基本功能
    (1).将输入数据作为其正向传播函数的参数
    (2).通过正向传播函数来生成输出(注意输入和输出的形状)
    (3).计算输出关于输入的梯度，可通过其反向传播函数进行访问。
    (4).存储和访问正向传播计算所需的参数
    (5).根据需要初始化模型参数|感觉比较重要，初始化方式的不同可能会导致训练时的不同情况|
'''
# 从零编写一个块，从父类继承，即只需要提供我们自己的构造函数(__init__)和正向传播函数

class MLP(nn.Module):
    # 声明两个全连接层
    def __init__(self):
        # 调用父类的构造函数，可以同时指定其他函数参数
        super().__init__()
        self.hidden = nn.Linear(20, 256) # 隐藏层
        self.out = nn.Linear(256, 10) # 输出层
    
    # 正向传播
    def forward(self, X):
        # Relu函数激活
        # hidden->relu->out
        return self.out(F.relu(self.hidden(X)))

net = MLP()
# print(net(X))

# 顺序块(nn.Sequential()的工作内容)
# (1).将块逐个追加到列表中
# (2).正向传播函数，用来按顺序串联块
# 在__init__方法中，将每个块添加到||有序字典||_modules中，在块的参数初始化过程中，
# 系统可以知道在_modules字典中查找需要初始化参数的子块
class MySequential(nn.Module):
    def __init__(self, *args):
        # *args中存了一串块
        super().__init__()
        for block in args:
            # 取出各个块
            # block为Module子类的一个实例
            # self._modules = OrderedDict()
            self._modules[block] = block
    
    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历他们
        for block in self._modules.values():
            X = block(X)
        return X

# 就像使用 Sequential类时那样使用自定义的顺序块
net = MySequential(nn.Linear(20, 256),nn.ReLU(), nn.Linear(256, 10))
# print(net(X))

# 正向传播函数
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数
        # 该参数rand_weight在实例化时被随机初始化，之后为常量即不会参与训练
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和dot函数
        # torch.mm(mat1, mat2, out=None)对矩阵mat1和mat2相乘
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层。两个全连接层共享参数？
        X = self.linear(X)
        # 控制流
        # 个人自定义的运算纪衡到神经网络的计算流程中
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
net = FixedHiddenMLP()
# print(net(X))

# 混合搭配块
class NestMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)
    
    def forward(self, X):
        # 嵌套块
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMlp(), nn.Linear(16, 20), FixedHiddenMLP())
# print(chimera(X))

# 练习：实现一个块，以两个块为参数并返回正向传播中两个网络的串联输出
class TwoBlockNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = MLP()
        self.net2 = nn.Linear(10, 5)
    
    def forward(self, X):
        # return self.net2(self.net1(X))
        X = self.net2(self.net1(X))
        return X.sum()

twoBlock = TwoBlockNet()
# print(twoBlock(X))