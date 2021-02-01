import numpy as np
import matplotlib.pyplot as plt


# 真正的函数
def g(x):
    return 0.1 * (x ** 3 + x ** 2 + x)


# 随意准备一些向真正的函数加入了一点噪声的训练数据
train_x = np.linspace(-2, 2, 8)
train_y = g(train_x) + np.random.randn(train_x.size) * 0.05

# 绘图确认
x = np.linspace(-2, 2, 100)
plt.plot(train_x, train_y, 'o')
plt.plot(x, g(x), linestyle='dashed')
plt.ylim(-1, 2)
plt.show()
