import matplotlib.pyplot as plt
import numpy as np

nets = ['Resnet18','EfficientNetB0', 'SENet18']

# 准备绘图
plt.figure(figsize=(10, 8))

for net in nets:
    root = 'batch' + str(net) + '.txt'
    # 读取数据
    data = np.genfromtxt(root)

    # 打印原始数据的行数
    print("原始数据的行数:", data.shape[0])

    # 定义 structured array 的 dtype
    dtype = [('col' + str(i), float) for i in range(data.shape[1])]

    # 将数据转换为 structured array
    structured_data = np.core.records.fromarrays(data.T, dtype=dtype)

    # 根据最后三列排序
    sorted_data = np.sort(structured_data, order=['col23', 'col24', 'col25'])

    # 使用 numpy 的 split 函数根据最后三列分组
    unique_keys, indices = np.unique(sorted_data[['col23', 'col24', 'col25']], return_index=True, axis=0)
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
    learning_rates = averaged_data[:, -1]  # 倒数第一列：batch size
    regions = averaged_data[:, -6]         # 倒数第六列：region数

    learning_rates = np.round(learning_rates, decimals=5)

    # 创建一个新的等距横坐标映射
    # 例如，对于learning rate，您可以创建一个映射：{0.001: 1, 0.01: 2, 0.1: 3}
    learning_rate_mapping = {32: 1, 64:2, 128:3, 256:4, 512:5, 1024:6}

    # 将learning rate的值映射到新的等距值
    mapped_learning_rates = np.array([learning_rate_mapping[lr] for lr in learning_rates])
    # 绘制线条
    plt.plot(mapped_learning_rates, regions, label=f'{net}', marker='o', markersize=10, linewidth=3)

    # 设置横坐标的刻度为新的等距值，并将标签设置为原始的非等距值
    plt.xticks(list(learning_rate_mapping.values()), list(learning_rate_mapping.keys()))
    plt.xticks(fontsize=18)  # 横坐标轴刻度
    plt.yticks(fontsize=18)  # 纵坐标轴刻度
    # 添加图例
    plt.legend(fontsize=18)

# 添加标题和轴标签
plt.title('Average Regions vs. Batch Size', fontsize=22)
plt.xlabel('Batch Size', fontsize=20)
plt.ylabel('Average Regions', fontsize=20)

# 显示图表
plt.savefig('region_counts_vs_batch_size.png')
