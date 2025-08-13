import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams['figure.dpi'] = 300


nets = ['Resnet18']
# 读取数据

for net in nets:
    root = str(net) + '.txt'
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

    # 把averaged_data的后六列保存到文件中
    np.savetxt(str(net) + '_averaged_data.txt', averaged_data[:, 20:], fmt='%.6f')