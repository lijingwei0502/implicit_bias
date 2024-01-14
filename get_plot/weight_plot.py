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
weight_decay = averaged_data[:, -3]  # 倒数第四列：weight decay
regions = averaged_data[:, -5]      # 倒数第六列：region数
learning_rates = averaged_data[:, -2]  # 倒数第三列：learning rate
batch_sizes = averaged_data[:, -1]     # 倒数第二列：batch size

# 获取 learning rate 和 batch size 的组合
unique_combinations = np.unique(np.vstack([learning_rates, batch_sizes]).T, axis=0)

# 创建一个新的等距横坐标映射
# 例如，对于learning rate，您可以创建一个映射：{0.001: 1, 0.01: 2, 0.1: 3}
weight_decay_mapping = {1e-7: 1, 1e-6: 2, 1e-5: 3}
# 准备绘图
plt.figure(figsize=(10, 6))

# 为每个组合绘制一条线
for lr, bs in unique_combinations:
    # 筛选出当前组合对应的行
    mask = (learning_rates == lr) & (batch_sizes == bs)
    # 将weight decay的值映射到新的等距值
    mapped_weight_decay = np.array([weight_decay_mapping[wd] for wd in weight_decay[mask]])
    # 绘制线条
    plt.plot(mapped_weight_decay, regions[mask], label=f'LR: {lr}, BS: {bs}')

plt.xticks(list(weight_decay_mapping.values()), list(weight_decay_mapping.keys()))

# 添加图例
plt.legend()

# 添加标题和轴标签
plt.title('Region Counts vs. Weight Decay')
plt.xlabel('Weight Decay')
plt.ylabel('Region Counts')

# 显示图表
plt.savefig('result_corr/region_counts_vs_weight_decay.png')