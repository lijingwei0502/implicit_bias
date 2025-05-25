import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats

# 读取数据
data = np.genfromtxt('-5.txt')

# 打印原始数据的行数
print("原始数据的行数:", data.shape[0])

data = data[data[:, 0] == 2]

indexs = [21]

# 为每个 epoch 值绘制并保存一张散点图
for index in indexs:
    # 过滤出当前 epoch 的数据
    x_current = data[:, index]
    y_current = data[:, 22] - data[:, 23] 
    correlation_coefficient_pearson = np.corrcoef(x_current, y_current)[0, 1]  # 计算相关系数
    # 绘制散点图
    print(x_current.mean())
    plt.scatter(x_current, y_current)
    plt.xlabel('Region Counts')
    plt.ylabel('Generalization Gap')
    plt.title(f'Correlation: {correlation_coefficient_pearson:.2f}')
    plt.savefig('correlation.png')  # 保存图像
    plt.clf()  # 清除当前图像

