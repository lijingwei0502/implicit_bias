import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import MaxNLocator
# 读取数据

root = 'Resnet18.txt'
data = np.genfromtxt(root)
# 打印原始数据的行数
print("原始数据的行数:", data.shape[0])

averaged_data = data
#averaged_data = data[data[:,5] > 0.001+1e-7]
print("处理后数据的行数:", averaged_data.shape[0])

indexs = [0,1,2]
plt.figure(figsize=(10, 8))
for index in indexs:
    x_current = averaged_data[:, index]
    y_current = 100 - averaged_data[:,3]
    correlation = np.corrcoef(x_current, y_current)[0, 1]  # 计算相关系数
    ax = plt.gca()  # 获取当前轴对象
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))  # 强制y轴标签为整数，并限制最大刻度数为5
    if index == 0:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))  # 强制y轴标签为整数，并限制最大刻度数为5
    elif index == 1:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    # 绘制散点图
    plt.scatter(x_current, y_current,s=100)
    fontsize = 38
    if index == 0:
        plt.xlabel('$d(W)$',fontsize=fontsize)
    elif index == 1:
        plt.xlabel('$\gamma(W)$',fontsize=fontsize)
    elif index == 2:
        plt.xlabel('$\gamma(W)/d(W)$',fontsize=fontsize)
    #plt.ylabel('Generalization Gap')
    plt.ylabel('Generalization Gap',fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.grid(True)
    if index == 0:
        plt.title(f'Correlation of $d(W)$: {correlation:.2f}', fontsize=fontsize)
    elif index == 1:
        plt.title(f'Correlation of $\gamma(W)$: {correlation:.2f}', fontsize=fontsize)
    elif index == 2:
        plt.title(f'Correlation of $\gamma(W)/d(W)$: {correlation:.2f}', fontsize=fontsize)
    plt.subplots_adjust(left=0.22, right=0.85, top=0.85, bottom=0.20)  # 这里的数值可以根据需要调整
    root = str(index) + '.png'
    plt.savefig(root)  # 保存图像
    plt.clf()  # 清除当前图像

