import numpy as np
import matplotlib.pyplot as plt

# 读入训练数据
train = np.loadtxt('click.csv', delimiter=',', skiprows=1)
train_x = train[:, 0]
train_y = train[:, 1]

# 绘图
plt.plot(train_x, train_y, 'o')
plt.show()

# 5-2-3
# 标准化
mu = train_x.mean()
sigma = train_x.std()


def standardize(x):
    return (x - mu) / sigma


train_z = standardize(train_x)


# 预测函数  5-2-8
def f(x):
    return np.dot(x, theta)


# 目标函数  5-2-8
def E(x, y):
    return 0.5 * np.sum((y - f(x)) ** 2)


# 创建训练数据的矩阵
def to_matrix(x):
    return np.vstack([np.ones(x.shape[0]), x, x ** 2]).T


X = to_matrix(train_z)

# 学习率
ETA = 1e-3


# 均方误差
def MSE(x, y):
    return (1 / x.shape[0]) * np.sum((y - f(x)) ** 2)


# 用随机值初始化参数
theta = np.random.rand(3)

# 均方误差的历史记录
errors = []

# 误差的差值
diff = 1

# 重复学习
errors.append(MSE(X, train_y))

# 更新的次数
count = 0

while (diff > 1e-2) and (count < 500):
    theta = theta - ETA * np.dot(f(X) - train_y, X)
    errors.append(MSE(X, train_y))
    diff = errors[-2] - errors[-1]
    count += 1
    print(count)

# 绘制误差变化图
x = np.arange(len(errors))
plt.plot(x, errors)
plt.show()
