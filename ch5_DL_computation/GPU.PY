# 使用GPU 
import torch
from torch import nn
from torch.nn.functional import relu

print(torch.device('cpu'), torch.cuda.device('cuda'), torch.cuda.device('cuda:1'))

# 查询可用GPU数量
print(torch.cuda.device_count())

# 定义函数，允许在请求GPU不存在时运行代码

def try_gpu(i = 0):
    '''如果存在，返回gpu(i)，否则返回cpu()'''
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():
    '''返回所有可用的GPU，如果没有GPU，则返回[cpu()]。'''
    device = [torch.device(f'cuda:{i}')
        for i in range(torch.cuda.device_count())]
    return device if device else [torch.device('cpu')]

print(try_gpu(), try_gpu(10), try_all_gpus())

# 张量与GPU
# 默认张量在CPU上创建，查询张量所在设备
x = torch.tensor([1, 2, 3])
print(x.device) # cpu

# 存储在GPU上
X = torch.ones(2, 3, device=try_gpu())
print(X)

# 在第二个GPU上创建一个随机张量。
Y = torch.rand(2, 3, device=try_gpu(1))
print(Y)

# 复制
# 参数不在同一个GPU上时要将他们放在一块上计算
Z = X.cuda(1)
print(X)
print(Z)
print(Y)
'''
tensor([[1., 1., 1.],
        [1., 1., 1.]], device='cuda:0')
tensor([[1., 1., 1.],
        [1., 1., 1.]], device='cuda:1')
tensor([[0.1772, 0.5800, 0.3472],
        [0.2812, 0.6505, 0.4139]], device='cuda:1')
'''

# 同一块GPU上的参数相加
print(Y + Z)
# tensor([[1.1772, 1.5800, 1.3472],
#         [1.2812, 1.6505, 1.4139]], device='cuda:1')

# 神经网络与GPU
net = nn.net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
net(X)
# tensor([[-0.6285],
#       [-0.6285]], device='cuda:0', grad_fn=<AddmmBackward>)

# 确认模型参数在一个GPU上
print(net[0].weight.data.device)
# device(type='cuda', index=0)