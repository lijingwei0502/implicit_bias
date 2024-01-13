import matplotlib.pyplot as plt
import numpy as np

# 读取数据
data = np.genfromtxt('resulttxt/resultbatchres18.txt', usecols=range(25))

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
batch_sizes = averaged_data[:, -1]  # 倒数第二列：batch size
regions = averaged_data[:, -5]      # 倒数第六列：region数
learning_rates = averaged_data[:, -2]  # 倒数第三列：learning rate
weight_decays = averaged_data[:, -3]   # 倒数第四列：weight decay

# 获取 learning rate 和 weight decay 的组合
unique_combinations = np.unique(np.vstack([learning_rates, weight_decays]).T, axis=0)

# 对于batch size，您可以创建一个映射：{256: 1, 512: 2, 1024: 3}
batch_size_mapping = {8: 1, 16: 2, 32: 3, 64: 4, 128: 5, 256: 6, 512: 7, 1024: 8, 2048: 9}
# 准备绘图
plt.figure(figsize=(10, 8))

# 为每个组合绘制一条线
for lr, wd in unique_combinations:
    # 筛选出当前组合对应的行
    mask = (learning_rates == lr) & (weight_decays == wd)
    mapped_batch_sizes = np.array([batch_size_mapping[bs] for bs in batch_sizes[mask]])
    # 绘制线条
    plt.plot(mapped_batch_sizes, regions[mask], label=f'LR: {lr}, WD: {wd}')

# 设置横坐标的刻度为新的等距值，并将标签设置为原始的非等距值
plt.xticks(list(batch_size_mapping.values()), list(batch_size_mapping.keys()))

# 添加图例
plt.legend()

# 添加标题和轴标签
plt.title('Region Counts vs. Batch Size')
plt.xlabel('Batch Size')
plt.ylabel('Region Counts')

# 显示图表
plt.savefig('result_corr/region_counts_vs_batch_size.png')
