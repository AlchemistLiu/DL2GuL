import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256
train_iter, tese_iter = d2l.load_data_fashion_mnist(batch_size)

# 图像原格式为1*28*28，将图像拉长
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition

# softmax回归模型
def net(X):
    return softmax(torch.matmul((X.reshape(-1, W.shape[0])), W) + b)


# 交叉熵损失
def cross_entropy(y_hat, y ):
    return -torch.log(y_hat[range(len(y_hat)), y])

# 预测与真实比较
def accuracy(y_hat, y):
    # 确定是二维向量
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis = 1)
    # 比较数据类型
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

# 评估准确率
def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()  # 评估模式
    metric = Accumulator(2)
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
        return metric[0] / metric[1]

class Accumulator:

    def __init__(self, n):
        self.data = [0.0] * n
    
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# softmax回归训练

def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()

    metric = Accumulator(3)

    for X , y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizerp):
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(
                float(l) * len(y), accuracy(y_hat, y),
                y.size().numel())
        else:
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numle())
    return metric[0] / metric[2], metric[1] / metric[2]