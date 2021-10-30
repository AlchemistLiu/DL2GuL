# 参数初始化
# 参数绑定
import torch
from torch import nn
from torch.nn.init import xavier_normal_

# 单隐藏层的多层感知机
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand((2, 4))
# print(net(X))


# 调用内置初始化器
# 将所有权重参数初始化为标准差为0.01的高斯随机变量
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
net.apply(init_normal)
# print(net[0].weight.data[0], net[0].bias.data[0])
# tensor([ 0.0118,  0.0086, -0.0064, -0.0002]) tensor(0.)

# 将所有参数初始化为给定的参数
# constant_
def init_constand(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
net.apply(init_constand)
print(net[0].weight.data[0], net[0].bias.data[0])
# tensor([1., 1., 1., 1.]) tensor(0.)

# Xavier初始化
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
# 第二次全变成42
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(xavier)
net[2].apply(init_42)
# print(net[0].weight.data[0])
# print(net[2].weight.data)

# 自定义初始化
def my_init(m):
    if type(m) == nn.Linear:
        print('Init',
        *[(name, param.shape) for name, param in m.named_parameters()][0]) # [0]表示初始化的weight
        # U(10,10)的均匀分布
        nn.init.uniform_(m.weight, -10, 10)
        # c *= a 等效于 c = c * a
        # 将小于5的值置0
        # m.weigh.data.abs() >= 5 返回Bool类型
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
# 八个神经元对应八组weight
print(net[0].weight[:2]) 
'''
Init weight torch.Size([8, 4])
Init weight torch.Size([1, 8])
tensor([[-0.0000, -0.0000, -7.8156, -5.7044],
        [ 0.0000, -0.0000, -0.0000, -6.8014]], grad_fn=<SliceBackward>)
'''

# 直接设置参数
# weight加一
net[0].weight.data[:] += 1
# print(net[0].weight.data)
# 第一个神经元的第一个weight设为42
net[0].weight.data[0, 0] = 42
# print(net[0].weight.data) 

# 参数绑定
# 多个层间共享参数，定义一个稠密层，用他的参数来设置另一个层的参数
# 定义共享层名称
share = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    share, nn.ReLU(),
                    share, nn.ReLU(),
                    nn.Linear(8, 1))
# net(X)
# 检查参数是否相同
# print(net[2].weight.data[0] == net[4].weight.data[0])
# net[4].weight.data[0, 0] = 100 
# print(net[2].weight.data[0] == net[4].weight.data[0])




'''
# 进行训练，观察各层参数和梯度
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
net.apply(xavier)
for i in range(5):
    l = loss(net(X), y)
    trainer.zero_grad()
    l.backward()
    trainer.step()
    l = loss(net(X), y)
    print(l)
print(net[0].weight.grad)
print(net[2].weight.grad)
print(net[4].weight.grad)
print(net[6].weight.grad)
print(net[0].weight.data[0])
print(net[2].weight.data[0])
print(net[4].weight.data[0])
print(net[6].weight.data[0])
'''