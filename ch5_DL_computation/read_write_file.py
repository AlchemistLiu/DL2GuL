# 加载和保存张量
import torch 
from torch import nn
from torch.nn import functional as F

# 通过load和save的使用完成
X = torch.arange(4)
torch.save(X, 'X-file')
# print(X)
X2 = torch.load('X-file')
# print(X2)

# 存储一个张量列表并读取
y = torch.zeros(4)
torch.save([X, y], 'X-files')
X2, y2 = torch.load('X-files')
# print((X2, y2))

# 写入或读取从字符串映射到张量的字典
mydict = {'X': X, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
print(mydict2)
# {'X': tensor([0, 1, 2, 3]), 'y': tensor([0., 0., 0., 0.])}

# 加载和保存模型参数
# 简单多层感知机
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, X):
        return self.output(F.relu(self.hidden(X)))

net = MLP()
X = torch.randn(size=(2,20))
Y = net(X)
# 保存模型参数
torch.save(net.state_dict(), 'mlp.params')

# 恢复模型
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
print(clone.eval())
# MLP(
#   (hidden): Linear(in_features=20, out_features=256, bias=True)
#   (output): Linear(in_features=256, out_features=10, bias=True)
# )

# 验证两个模型参数是否相同
Y_clone = clone(X)
print(Y_clone == Y)
# tensor([[True, True, True, True, True, True, True, True, True, True],
#         [True, True, True, True, True, True, True, True, True, True]])