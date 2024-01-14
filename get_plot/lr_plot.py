import matplotlib.pyplot as plt
import numpy as np
import os

os.chdir(os.path.dirname(os.getcwd()))
# 读取数据
data = np.genfromtxt('single_corr/resnet18.txt', usecols=range(25))

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
# 提取所需的列
learning_rates = averaged_data[:, -2]  # 倒数第三列：learning rate
regions = averaged_data[:, -5]         # 倒数第六列：region数
batch_sizes = averaged_data[:, -1]     # 倒数第二列：batch size
weight_decays = averaged_data[:, -3]   # 倒数第四列：weight decay

learning_rates = np.round(learning_rates, decimals=5)

# 获取 batch size 和 weight decay 的组合
unique_combinations = np.unique(np.vstack([batch_sizes, weight_decays]).T, axis=0)

# 创建一个新的等距横坐标映射
# 例如，对于learning rate，您可以创建一个映射：{0.001: 1, 0.01: 2, 0.1: 3}
learning_rate_mapping = {0.001: 1, 0.002:2, 0.003:3, 0.004:4, 0.005:5, 0.006:6, 0.007:7, 0.008:8, 0.009:9, 0.01: 10, 0.02: 11, 0.03: 12, 0.04: 13, 0.05: 14, 0.06: 15, 0.07: 16, 0.08: 17, 0.09: 18, 0.1: 19}

# 准备绘图
plt.figure(figsize=(10, 8))

# 为每个组合绘制一条线
for bs, wd in unique_combinations:
    # 筛选出当前组合对应的行
    mask = (batch_sizes == bs) & (weight_decays == wd)
    # 将learning rate的值映射到新的等距值
    mapped_learning_rates = np.array([learning_rate_mapping[lr] for lr in learning_rates[mask]])
    # 绘制线条
    plt.plot(mapped_learning_rates, regions[mask], label=f'BS: {bs}, WD: {wd}')

# 设置横坐标的刻度为新的等距值，并将标签设置为原始的非等距值
plt.xticks(list(learning_rate_mapping.values()), list(learning_rate_mapping.keys()))

# 添加图例
plt.legend()

# 添加标题和轴标签
plt.title('Region Counts vs. Learning Rate')
plt.xlabel('Learning Rate')
plt.ylabel('Region Counts')

# 显示图表
plt.savefig('result_corr/region_counts_vs_learning_rate.png')
