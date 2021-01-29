# 尝试线性不可分的问题
import numpy as np
import matplotlib.pyplot as plt

# 读入训练数据
train = np.loadtxt('data3.csv', delimiter=',', skiprows=1)
train_x = train[:, 0:2]
train_y = train[:, 2]

# plt.plot(train_x[train_y == 1, 0], train_x[train_y == 1, 1], 'o')
# plt.plot(train_x[train_y == 0, 0], train_x[train_y == 0, 1], 'x')
# plt.show()

# 初始化参数 5-4-8
theta = np.random.rand(4)

# 标准化
def standardize(x):
    mu = x.mean()
    sigma = x.std(axis=0)
    return (x - mu) / sigma


train_z = standardize(train_x)


# 增加x0,x3
def to_matrix(x):
    x0 = np.ones([x.shape[0], 1])
    x3 = x[:, 0, np.newaxis] ** 2
    return np.hstack([x0, x, x3])


X = to_matrix(train_z)


# sigmoid函数
def f(x):
    return 1 / (1 + np.exp(-np.dot(x, theta)))


# 学习率
ETA = 1e-3

# 重复次数
epoch = 5000

# 重复学习
for _ in range(epoch):
    theta = theta - ETA * np.dot(f(X) - train_y, X)

# 画图 5-4-10
x1 = np.linspace(-2, 2, 100)
x2 = -(theta[0] + theta[1] * x1 + theta[3] * x1 ** 2) / theta[2]

plt.plot(train_z[train_y == 1, 0], train_z[train_y == 1, 1], 'o')
plt.plot(train_z[train_y == 0, 0], train_z[train_y == 0, 1], 'x')
plt.plot(x1, x2, linestyle='dashed')
plt.show()
