# 卷积神经网络（LeNet）
import torch
from torch import nn
from d2l import torch as d2l
from matplotlib import pyplot as plt
# 调整输入格式
class Reshape(torch.nn.Module):
    def forward(self, x):
        # -1表示批量大小不变
        return x.view(-1, 1, 28, 28)

# 网络定义
net = torch.nn.Sequential(
    Reshape(),
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
    nn.Linear(120, 84), nn.ReLU(),
    nn.Linear(84, 10))
'''
net = torch.nn.Sequential(
    Reshape(),
    nn.Conv2d(1, 6, kernel_size=3, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 10, kernel_size=3), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    # Flatten output shape:    torch.Size([1, 360])
    nn.Linear(360, 180), nn.ReLU(),
    nn.Linear(180, 84), nn.ReLU(),
    nn.Linear(84, 10))


net = torch.nn.Sequential(
    Reshape(), nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),# 第一层卷积
    nn.AvgPool2d(kernel_size=2, stride=2), # 池化1
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),# 第二卷积层
    nn.AvgPool2d(kernel_size=2, stride=2),# 池化2 这里都是2可以不用写stride
    nn.Flatten(),# 展平
    nn.Linear(16*5*5, 120), nn.Sigmoid(),# 全连接1
    nn.Linear(120, 84), nn.Sigmoid(),# 全连接2
    nn.Linear(84, 10))
'''
'''
# 检查模型
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
Reshape output shape:    torch.Size([1, 1, 28, 28])
Conv2d output shape:     torch.Size([1, 6, 28, 28])
Sigmoid output shape:    torch.Size([1, 6, 28, 28])
AvgPool2d output shape:  torch.Size([1, 6, 14, 14]) 
Conv2d output shape:     torch.Size([1, 16, 10, 10])
Sigmoid output shape:    torch.Size([1, 16, 10, 10])
AvgPool2d output shape:  torch.Size([1, 16, 5, 5])
Flatten output shape:    torch.Size([1, 400])
Linear output shape:     torch.Size([1, 120])
Sigmoid output shape:    torch.Size([1, 120])
Linear output shape:     torch.Size([1, 84])
Sigmoid output shape:    torch.Size([1, 84])
Linear output shape:     torch.Size([1, 10])
'''

# 模型训练
batch_size = 256
# 用fashion_mnist来试试
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# 用GPU跑了
# device需要传入指定的设备(gpu)
def evaluate_acuracy_gpu(net, data_iter, device=None):
    # 确定网络格式是否正确
    if isinstance(net, torch.nn.Module):
        # 测试时有这句 
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    # 累加器
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        # 将数据移到device
        if isinstance(X, list):
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        # metric accuracy  number(y)
        metric.add(d2l.accuracy(net(X), y), y.numel())
    # 返回data_iter中所有条目的data_iter / 条目数目
    return metric[0] / metric[1]

# 训练函数
# 同样添加指定设备
def train(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            # 均匀分布初始化
            nn.init.xavier_uniform_(m.weight)
        # 对init_weights中所有都初始化一次
    net.apply(init_weights)
    # 开run!
    print('training on', device)
    # 放GPU上
    net.to(device)
    # 选择优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=lr) 
    # 交叉熵损失
    loss = nn.CrossEntropyLoss()
    # 可视化结果
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    # 计时
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            # 梯度清零
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            # step(closure) [source]    进行单次优化 (参数更新).
            optimizer.step()
            # loss  acc number
            metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, 
                            (train_l, train_acc, None))
        # 测试集准确度
        test_acc = evaluate_acuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
        print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, 'f'test acc {test_acc:.3f}')
        print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec 'f'on {str(device)}')
# 开炼！
lr, num_epochs = 0.20, 10
train(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
plt.show()

'''
# 调试结果
1. 不改动网络结构。
    1. lr, num_epochs = 0.6, 10   loss 0.503, train acc 0.812, test acc 0.807。
    2. lr, num_epochs = 0.1, 10   loss 1.160, train acc 0.574, test acc 0.568。 感觉epoch小了，精度还能上升
    3. lr, num_epochs = 0.1, 20   loss 0.754, train acc 0.723, test acc 0.720。 可能要改结构了，调大一点学习率再试试
    4. lr, num_epochs = 0.2, 20   loss 0.567, train acc 0.783, test acc 0.751。 lr在0.1~0.2区间好像只要10个epoch就差不多了
    5. lr, num_epochs = 0.2, 10   loss 0.740, train acc 0.722, test acc 0.726。 看似好像lr大一点test acc高一点
    6. lr, num_epochs = 0.8, 10   loss 0.482, train acc 0.817, test acc 0.798。 雀食，再大一点
    7. lr, num_epochs = 0.9, 10   loss 0.469, train acc 0.822, test acc 0.788。 估计到这了，改下网络内参数试试（现在是经典LeNet）
2. 改动网络结构。
    1. 将sigmoid激活全部换成Relu,不改变通道等参数，适当调小lr
        1. lr, num_epochs = 0.10, 10   loss 0.340, train acc 0.874, test acc 0.855。 稍微有点过拟合，lr调大点
        2. lr, num_epochs = 0.15, 10   loss 0.322, train acc 0.881, test acc 0.864。 稍微好点，再调大点
        3. lr, num_epochs = 0.20, 10   loss 0.309, train acc 0.884, test acc 0.824。 感觉过拟合反而严重了
        4. lr, num_epochs = 0.01, 20   loss 0.450, train acc 0.836, test acc 0.792。 纯粹想试下0.01，调大一点num_epochs看看结果-->感觉这模型就这样了。
    2. 将sigmoid激活全部换成Relu,池化从avg改成max
        1. lr, num_epochs = 0.15, 10   loss 0.270, train acc 0.899, test acc 0.865。 效果好一点底，加大点lr
                                       loss 0.265, train acc 0.901, test acc 0.871
        2. lr, num_epochs = 0.20, 10   loss 0.279, train acc 0.895, test acc 0.880。 牺牲一点准确率，减少拟合
    3. 将sigmoid激活全部换成Relu,池化从avg改成max,论文里面6和16两个通道好像是最好的，就不变了，
    卷积核大小调到3这样第三层卷积后展平为1*360，接下来的层修改到360->180->84->10
        1. lr, num_epochs = 0.15, 10   loss 0.273, train acc 0.897, test acc 0.858。 不太行
'''
# 保存模型
# torch.save(net.state_dict(), r'F:\Code\d2l-zh\myDLstudy\ch6_CNN\NLeNet.params')