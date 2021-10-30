from matplotlib import rc_params_from_file
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

'''
line 53
'''

# 生成人工数据集
# 𝑦=0.05+∑𝑖=1𝑑0.01𝑥𝑖+𝜖 where 𝜖∼N(0,0.012)
'''
标签是关于输入的线性函数。
标签同时被均值为0，标准差为0.01高斯噪声破坏。为了使过拟合的效果更加明显，
我们可以将问题的维数增加到 𝑑=200 ，并使用一个只包含20个样本的小训练集。
'''
# 训练数据越小时，越容易过拟合（20/100）
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_W, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
# 随机生成n_train维度的X 
# train_data = X * true_W + true_b + noise
train_data = d2l.synthetic_data(true_W, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_W, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)

# 从零实现
# 初始化模型参数
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

# L2范数惩罚
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2 #除二方便求导

# 训练代码
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss 
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log', 
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            #with torch.enable_grad():
            # 增加了L2范数惩罚项，广播机制使l2_penalty(w)成为一个长度为`batch_size`的向量
            # L(w,b)+λ2*L2(w)
            l = loss(net(X), y) + lambd * l2_penalty(w)
            # ___________________________________________
            l.sum().backward()
            # 有点问题
            # ___________________________________________
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                    d2l.evaluate_loss(net, test_iter, loss)))
        print('w的L2范数是：', torch.norm(w).item())

# 使用lambd = 0 禁用权重衰减后运行 
# train(lambd=0)
# 这里训练误差有了减少，但测试误差没有减少。这意味着出现了严重的过拟合
# 使用权重衰减来运行代码
# train(lambd=3)
# 训练误差增大，但测试误差减小。是期望从正则化中得到的效果
# plt.show()

# 简易实现
def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
        loss = nn.MSELoss()
        num_epochs, lr = 100, 0.003
        trainer = torch.optim.SGD([{"params": net[0].weight,
                                    "weight_decay": wd}, {
                                    "params": net[0].bias}], 
                                    lr = lr)
        animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log', 
                            xlim=[5, num_epochs], legend=['train', 'test'])
        for epoch in range(num_epochs):
            for X, y in train_iter:
                # with torch.enable_grad():
                trainer.zero_grad()
                l = loss(net(X), y)
                l.backward()
                trainer.step()
            if (epoch + 1) % 5 == 0:
                animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                        d2l.evaluate_loss(net, test_iter, loss)))
            print('w的L2范数是：', net[0].weight.norm().item())

# 禁用权重衰减
# train_concise(0)
# 使用权重衰减来运行代码
# train_concise(3)
# plt.show()
