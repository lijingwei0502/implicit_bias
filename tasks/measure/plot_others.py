import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams['figure.dpi'] = 300



root = 'measure.txt'
data = np.genfromtxt(root)

# 打印原始数据的行数
print("原始数据的行数:", data.shape[0])


averaged_data = data
# 打印处理后数据的行数
print("处理后数据的行数:", averaged_data.shape[0])

#indexs = [1,5,10,15,20]
indexs = [3,4,5,6,7,8,9]
plt.figure(figsize=(10, 8))
for index in indexs:
    # 过滤出当前 epoch 的数据
    x_current = averaged_data[:, index]
    y_current = averaged_data[:, 10]-averaged_data[:, 11]
    correlation = np.corrcoef(x_current, y_current)[0, 1]  # 计算相关系数
    coeffs = np.polyfit(x_current, y_current, 1)
    regression_line = np.poly1d(coeffs)
    x_range = np.linspace(min(x_current), max(x_current), 100)
    plt.plot(x_range, regression_line(x_range), label=None, linestyle='--', linewidth=3.5)
    # 绘制散点图
    plt.scatter(x_current, y_current)
    plt.xlabel('Region Counts',fontsize=34)
    plt.ylabel('Generalization Gap',fontsize=34)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=34)
    root = 'correlation.png'
    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)  # 这里的数值可以根据需要调整
    plt.savefig(root)  # 保存图像
    plt.clf()  # 清除当前图像
    print(f'Correlation between Region Counts and Generalization Gap: {correlation:.4f}')
    


