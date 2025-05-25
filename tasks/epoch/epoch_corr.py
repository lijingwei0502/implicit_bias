import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats

# 读取数据
data = np.genfromtxt('epoch.txt')

# 打印原始数据的行数
print("原始数据的行数:", data.shape[0])


indexs = [0,1,2,3,4,5,6,7]
x_all = []
y_all = []
# 为每个 epoch 值绘制并保存一张散点图
for index in indexs:
    # 过滤出当前 epoch 的数据
    x_current = data[:, index]
    y_current = data[:, index + 8] - data[:, index + 16]
    x_mean = np.mean(x_current, axis=0)
    y_mean = np.mean(y_current, axis=0)
    x_var = np.var(x_current, axis=0)
    y_var = np.var(y_current, axis=0)
    correlation_coefficient_pearson = np.corrcoef(x_current, y_current)[0, 1]  # 计算相关系数
    print(index)
    print(x_mean)
    print(y_mean)
    print(x_var)
    print(y_var)
    print(correlation_coefficient_pearson)
    print()
    # 绘制散点图
    plt.scatter(x_current, y_current)
    plt.xlabel('Average Region')
    plt.ylabel('Generalization Gap')
    plt.title(f'Correlation: {correlation_coefficient_pearson:.2f}')
    plt.savefig(str(index) + 'epoch_correlation.png')  # 保存图像
    plt.clf()  # 清除当前图像

