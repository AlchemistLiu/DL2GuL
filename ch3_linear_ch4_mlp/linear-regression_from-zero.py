import random
import torch
from d2l import torch as d2l

#生成数据集
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples,len(w)))
    y = torch.matmul(X, w) + b
    #添加噪声
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1,1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2

features, labels = synthetic_data(true_w, true_b, 1000)


# 读取数据集

def data_iter(batch_size, features, labels):
    #打乱抓取数据
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)

    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i : min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]



#线性模型
def linreg(X, w, b):
    return torch.matmul(X, w) + b

#损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 /2

#随机梯度下降
def sdg(params, lr, batch_size):
    #不进行反向传播，不需要保存计算图，即no_gard
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

#初始化模型参数
w = torch.normal(0, 0.01, size=(2, 1 ), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

#开始训练
lr = 0.03
num_epochs = 6
batch_size = 10
net =linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sdg([w, b], lr, batch_size)
    
    with torch.no_grad():
        train_l = loss(net(features, w, b),labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')