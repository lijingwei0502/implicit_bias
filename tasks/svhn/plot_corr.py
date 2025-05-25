import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams['figure.dpi'] = 300


nets = ['Resnet18','Resnet34', 'VGG19', 'MobileNet', 'SENet18', 'ShuffleNetV2', 'EfficientNetB0', 'RegNetX_200MF', 'SimpleDLA']
# 读取数据

for net in nets:
    root = str(net) + '_svhn.txt'
    if not os.path.exists(root):
        continue
    data = np.genfromtxt(root)

    # 打印原始数据的行数
    print("原始数据的行数:", data.shape[0])

    # 定义 structured array 的 dtype
    dtype = [('col' + str(i), float) for i in range(data.shape[1])]

    # 将数据转换为 structured array
    structured_data = np.core.records.fromarrays(data.T, dtype=dtype)

    # 根据最后三列排序
    sorted_data = np.sort(structured_data, order=['col24', 'col25', 'col26'])

    # 使用 numpy 的 split 函数根据最后三列分组
    unique_keys, indices = np.unique(sorted_data[['col24', 'col25', 'col26']], return_index=True, axis=0)
    grouped_data = np.split(sorted_data, indices[1:])

    # 对每个组进行平均
    averaged_data = []
    for group in grouped_data:
        # 将每个分组转换为普通的 NumPy 数组
        group_array = np.array(group.tolist())
        averaged_data.append(group_array.mean(axis=0))

    averaged_data = np.array(averaged_data)
    # 打印处理后数据的行数
    print("处理后数据的行数:", averaged_data.shape[0])

    # 去掉第25列=0.1的数据
    averaged_data = averaged_data[averaged_data[:, 25] != 0.1]

    #indexs = [1,5,10,15,20]
    indexs = [10]
    plt.figure(figsize=(10, 8))
    for index in indexs:
        # 过滤出当前 epoch 的数据
        x_current = averaged_data[:, index]
        y_current = averaged_data[:, 22]-averaged_data[:, 23]
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
        plt.title(f'Correlation of {net} : {correlation:.2f}', fontsize=34)
        root = str(net) + '.png'
        plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)  # 这里的数值可以根据需要调整
        plt.savefig(root)  # 保存图像
        plt.clf()  # 清除当前图像
        f = open('correlation.txt', 'a')
        f.write(str(net) + ' ' + str(correlation) + '\n')
        # 调整边距
    

