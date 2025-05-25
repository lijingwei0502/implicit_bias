import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats

nets = ['Resnet18','Resnet34','SimpleDLA','VGG19','EfficientNetB0','RegNetX_200MF', 'MobileNet', 'SENet18', 'EfficientNetB0']
for net in nets:
    # 读取数据
    if os.path.exists(f'{net}.txt'):
        all_data = np.genfromtxt(f'{net}.txt')

        indexs = [3,4,5,6]

        # 为每个 epoch 值绘制并保存一张散点图
        for index in indexs:
            data = all_data[all_data[:, 0] == index]
            # 过滤出当前 epoch 的数据
            x_current = data[:, 21]
            y_current = data[:, 22] - data[:,23] 
            correlation_coefficient_pearson = np.corrcoef(x_current, y_current)[0, 1]  # 计算相关系数
            # 绘制散点图
            plt.scatter(x_current, y_current)
            plt.xlabel('Average Region')
            plt.ylabel('Generalization Gap')
            plt.title(f'Correlation: {correlation_coefficient_pearson:.2f}')
            file_path = f'correlation/{net}_{index}.png'
            plt.savefig(file_path)  # 保存图像
            plt.clf()  # 清除当前图像

