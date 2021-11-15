import torch
from torch import nn
from d2l import torch as d2l
from matplotlib import pyplot as plt

class Reshape(torch.nn.Module):
    def forward(self, x):
        # -1表示批量大小不变
        return x.view(-1, 1, 28, 28)


net = torch.nn.Sequential(
    Reshape(), nn.Conv2d(1, 6, kernel_size=3, padding=1), nn.ReLU(),# 第一层卷积
    nn.AvgPool2d(kernel_size=3, padding=1, stride=2), # 池化1
    nn.Conv2d(6, 12, kernel_size=3, padding=1, stride=2), nn.ReLU(),# 第二卷积层
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),# 展平
    nn.Linear(108, 10))

X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
'''
Reshape output shape:    torch.Size([1, 1, 28, 28])
Conv2d output shape:     torch.Size([1, 6, 28, 28])
ReLU output shape:       torch.Size([1, 6, 28, 28])
AvgPool2d output shape:  torch.Size([1, 6, 14, 14])
Conv2d output shape:     torch.Size([1, 12, 7, 7])
ReLU output shape:       torch.Size([1, 12, 7, 7])
AvgPool2d output shape:  torch.Size([1, 12, 3, 3])
Flatten output shape:    torch.Size([1, 108])
Linear output shape:     torch.Size([1, 10])
'''

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

# 保存模型
# torch.save(net.state_dict(), r'myDLstudy\DLmeet\CNN.params')