# Q:卷积的输出形状取决于输入形状和卷积核的形状?
# A:填充（padding）和 步幅 (stride)需要被考虑
import torch 
from torch import nn


# 观察填充后输入和输入维度的变换，省略前两个维度和批量大小的通道
# 本质是一个比较函数
def comp_conv2d(conv2d, X):
    # 暂时不考虑通道和批量，故为(1,1)
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # 省略前两个维度，只比较卷积后的宽高
    return Y.reshape(Y.shape[2:])

# 一个卷积层
# ******************************************
# 卷积核大小尽量用奇数(暂时不知道为啥，先用着)
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
# padding = 1并不是只填充一行/列-->☞指每边都填充了1行或1列，因此总共添加了2行或2列
X = torch.rand(size=(8, 8))     # 8*8的输入
result = comp_conv2d(conv2d, X).shape
print(result)   # torch.Size([8, 8])
# (𝑛ℎ−𝑘ℎ+𝑝ℎ+1)×(𝑛𝑤−𝑘𝑤+𝑝𝑤+1) --> (8-3+2+1)×(8-3+2+1) -->8×8

# 填充不同的高度和宽度(2, 1)和不同的宽高的卷积核(5, 3)
conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
result = comp_conv2d(conv2d, X).shape
print(result)    # torch.Size([8, 8])
# (8-5+4+1)×(8-3+2+1) -->8×8

# 步幅strid对比
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
result = comp_conv2d(conv2d, X).shape
print(result)    # torch.Size([8, 8])
# ⌊(𝑛ℎ−𝑘ℎ+𝑝ℎ+𝑠ℎ)/𝑠ℎ⌋ × ⌊(𝑛𝑤−𝑘𝑤+𝑝𝑤+𝑠𝑤)/𝑠𝑤⌋. --> (8-3+2+2)/2 × (8-3+2+2)/2 --> 4×4

# 不同的高度和宽度上的步幅(3, 4)和不同的宽高的卷积核(3, 5)
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
result = comp_conv2d(conv2d, X).shape
print(result)    # torch.Size([2, 2])
# (8-3+0+3)/3×(8-5+2+4)/4 -->2×2