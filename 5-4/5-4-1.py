"""
逻辑回归的实现，首先要初始化参数，然后对训练数据标准化，x1,和x2要分别标准化，另外要增加一个x0列
"""

import numpy as np
import matplotlib.pyplot as plt

# 读入训练数据
train = np.loadtxt('images2.csv', delimiter=',', skiprows=1)
train_x = train[:, 0:2]
train_y = train[:, 2]

# 初始化参数
theta = np.random.rand(3)

# 标准化
mu = train_x.mean(axis=0)
sigma = train_x.std(axis=0)


def standardize(x):
    return (x - mu) / sigma


train_z = standardize(train_x)


# 增加 x0
def to_matrix(x):
    x0 = np.ones([x.shape[0], 1])
    return np.hstack([x0, x])


X = to_matrix(train_z)

# 将标准化后的训练数据画成图
plt.plot(train_z[train_y == 1, 0], train_z[train_y == 1, 1], 'o')
plt.plot(train_z[train_y == 0, 0], train_z[train_y == 0, 1], 'x')
plt.show()


# 下一个要做的是预测函数的实现，使用表达式3.5.2见过的sigmoid函数
# sigmoid函数
def f(x):
    return 1 / (1 + np.exp(-np.dot(x, theta)))

# 5-4-3
# 学习率
ETA = 1e-3

# 重复次数
epoch = 5000

# 重复学习
for _ in range(epoch):
    theta = theta - ETA * np.dot(f(X) - train_y,X)
# 用图来表示 5-4-4
x0 = np.linspace(-2, 2, 100)

plt.plot(train_z[train_y == 1, 0], train_z[train_y == 1, 1], 'o')
plt.plot(train_z[train_y == 0, 0], train_z[train_y == 0, 1], 'x')
plt.plot(x0, -(theta[0] + theta[1] * x0)  / theta[2], linestyle='dashed')
plt.show()

# 验证,使用下面语句在Console中执行看结果
# print(f(to_matrix(standardize([[200, 100], [100, 200]]))))


# 结果进行阈值设置
def classify(x):
    return (f(x) >= 0.5).astype(np.int)

# 验证,使用下面语句在Console中执行看结果
# classify(to_matrix(standardize([[200, 100], [100, 200]])))
