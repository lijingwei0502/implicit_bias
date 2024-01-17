import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats

# 读取数据
data = np.genfromtxt('resulttxt/resultlrres18.txt', usecols=range(25))

# 打印原始数据的行数
print("原始数据的行数:", data.shape[0])

# 定义 structured array 的 dtype
dtype = [('col' + str(i), float) for i in range(data.shape[1])]

# 将数据转换为 structured array
structured_data = np.core.records.fromarrays(data.T, dtype=dtype)

# 根据最后三列排序
sorted_data = np.sort(structured_data, order=['col22', 'col23', 'col24'])

# 使用 numpy 的 split 函数根据最后三列分组
unique_keys, indices = np.unique(sorted_data[['col22', 'col23', 'col24']], return_index=True, axis=0)
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


# 定义 region 数的最小阈值
region_threshold = 4
accuracy_threshold = 62

# 过滤掉 region 数较小的行
filter_condition = ~(np.isclose(averaged_data[:, 23], 0.5) | np.isclose(averaged_data[:, 23], 0.05) | np.isclose(averaged_data[:, 23], 0.005))

# 应用过滤条件
averaged_data = averaged_data[filter_condition]

# 打印过滤后数据的行数
print("过滤后数据的行数:", averaged_data.shape[0])

indexs = [1,5,10,15,20]
# 为每个 epoch 值绘制并保存一张散点图
for index in indexs:
    # 过滤出当前 epoch 的数据
    x_current = averaged_data[:, index]
    y_current = averaged_data[:, 21]

    correlation_coefficient_pearson = np.corrcoef(x_current, y_current)[0, 1]  # 计算相关系数

    # 绘制散点图
    plt.scatter(x_current, y_current)
    plt.xlabel('Average Region')
    plt.ylabel('Test Accuracy')
    plt.title(f'Epoch {index*10} - Correlation: {correlation_coefficient_pearson:.2f}')
    #plt.title(f'Epoch {index*10} - Linear Correlation: {correlation_coefficient_pearson:.2f}, Spearman Correlation: {correlation_coefficient_spearman:.2f}')
    if os.path.exists('result_corr') == False:
        os.mkdir('result_corr')
    plt.savefig(f'result_corr/correlation_epoch_{index*10}.png')  # 保存图像
    plt.clf()  # 清除当前图像

