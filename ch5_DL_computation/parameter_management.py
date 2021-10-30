# 访问参数
# 不同模型组件间共享参数
import torch
from torch import nn

# 单隐藏层的多层感知机
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand((2, 4))
# print(net(X))

# 查看第二个全连接层的参数
# **************************** #
# print(net[2].state_dict())
# **************************** #
# 从第二个神经网络层提取偏置


print(type(net[2].bias))   # <class 'torch.nn.parameter.Parameter'>
print(net[2].bias)         # tensor([0.0222], requires_grad=True)
print(net[2].bias.data)    # tensor([0.0222])
# 访问每个参数的梯度
print(net[2].weight.grad)  # None
# 访问第一个全连接层的参数和访问所有层
# 方法一
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
# ('weight', torch.Size([8, 4])) ('bias', torch.Size([8]))
print(*[(name, param.shape) for name, param in net.named_parameters()])
# ('0.weight', torch.Size([8, 4])) ('0.bias', torch.Size([8])) ('2.weight', torch.Size([1, 8])) ('2.bias', torch.Size([1]))
# 方法二
print(net.state_dict()['2.bias'].data)    # tensor([-0.0460])


# 从嵌套块收集参数
# 生成块
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                        nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block{i}', block1())
    return net
rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
print(rgnet(X))
# 查看网络组织顺序
print(rgnet)
'''
Sequential(
  (0): Sequential(
    (block0): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block1): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block2): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block3): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
  )
  (1): Linear(in_features=4, out_features=1, bias=True)
)
'''
# 访问嵌套层-》嵌套列表索引
# 访问第一个主块中第二个子块的第一层的b
print(rgnet[0][1][0].bias.data)