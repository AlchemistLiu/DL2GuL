# Dropout
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt


# 从零实现
# 输入X张量，以dropout的概率随机丢弃X中的元素
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1 
    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    # 随机生成矩阵，与dropout相比，大于将矩阵相应元素置1，小于置0
    mask = (torch.rand(X.shape) > dropout).float()
    # ℎ′=ℎ/1−𝑝 概率为 𝑝    其他情况直接丢弃（置0）
    # 此方法保证X的期望 𝐸[x′]=x 。
    return mask * X / (1.0 - dropout)

# 测试dropout_layer

'''
___________________________________________________________
X = torch.arange(16, dtype=torch.float32).reshape((2, 8))
print(X)
print(dropout_layer(X, 0))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1))
___________________________________________________________
'''
# 引入Fashion-MNIST数据集。定义具有两个隐藏层的多层感知机，每个隐藏层包含256个单元
num_inputs, num_outputs, num_hidden1, num_hidden2 = 784, 10, 256, 256
dropout_1, dropout_2 = 0.2, 0.5

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden1, num_hidden2,
                is_training=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hidden1)
        self.lin2 = nn.Linear(num_hidden1, num_hidden2)
        self.lin3 = nn.Linear(num_hidden2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 控制训练与测试时dropout层的使用
        if self.training:
            H1 = dropout_layer(H1, dropout_1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, dropout_2)
        out = self.lin3(H2)
        return out

# net = Net(num_inputs, num_outputs, num_hidden1, num_hidden2)

# 训练
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# trainer = torch.optim.SGD(net.parameters(), lr=lr)
# d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs,trainer)
# plt.show()



    
# 简洁实现
net = nn.Sequential(nn.Flatten(), 
                    # 隐藏层1 
                    nn.Linear(784, 256), nn.ReLU(),
                    # 第一个dropout层
                    nn.Dropout(dropout_1),
                    # 隐藏层2
                    nn.Linear(256, 256), nn.ReLU(),
                    # 第二个dropout层
                    nn.Dropout(dropout_2),
                    nn.Linear(256, 10))
# 初始化参数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

# 训练
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
plt.show()