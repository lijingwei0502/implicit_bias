import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats
plt.rcParams['figure.dpi'] = 300

models = ['dt', 'rf']

x = []
y = []
for model in models:
    # 读取数据
    dir = model + '.txt'
    data = np.genfromtxt(dir)

    # 打印原始数据的行数
    print("原始数据的行数:", data.shape[0])

    # # # 只保留第二行数据为1的行
    # data = data[data[:, 0] < 4]
    indexs = [0]

    # 为每个 epoch 值绘制并保存一张散点图
    for index in indexs:
        # 过滤出当前 epoch 的数据
        x_current = data[:, index]
        y_current = (data[:, 1] - data[:, 2]) * 100
        correlation = np.corrcoef(x_current, y_current)[0, 1]  # 计算相关系数
        print(correlation)
        x.append(x_current)
        y.append(y_current)

plt.figure(figsize=(10, 8))
# Plotting the data with specified markers and adding transparency
plt.scatter(x[0], y[0], label='Decision Tree', color='blue', marker='o', alpha=0.7)
plt.scatter(x[1], y[1], label='Random Forest', color='orange', marker='d', alpha=0.7)

# Performing linear regression and plotting regression lines
for i in range(len(x)):
    coeffs = np.polyfit(x[i], y[i], 1)
    regression_line = np.poly1d(coeffs)
    x_range = np.linspace(min(x[i]), max(x[i]), 100)
    if i == 0:
        plt.plot(x_range, regression_line(x_range), linestyle='--', alpha=0.7, linewidth=3.5, color='blue')  # Set color to blue
    else:
        plt.plot(x_range, regression_line(x_range), linestyle='--', alpha=0.7, linewidth=3.5, color='orange')  # Set color to orange

plt.xlabel('Region Counts', fontsize=24)
plt.ylabel('Generalization Gap', fontsize=24)
plt.legend(fontsize=22)
plt.tick_params(axis='both', which='major', labelsize=22)
plt.grid(True)
plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
plt.savefig('ml.png')  # Saving the figure again
