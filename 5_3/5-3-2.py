import numpy as np
import matplotlib.pyplot as plt

# 权重的初始化
w = np.random.rand(2)

# 判别函数
def f(x):
    if np.dot(w,x) >= 0:
        return 1
    else:
        return -1