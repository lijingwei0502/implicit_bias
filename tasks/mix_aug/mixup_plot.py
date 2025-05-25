import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.dpi'] = 300
# 读取数据

datasets = ['primal', 'mixup']
x = []
y = []
for dataset in datasets:
    root =  str(dataset) + '.txt'
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

    #indexs = [1,5,10,15,20]
    indexs = [20]
    # 为每个 epoch 值绘制并保存一张散点图
    for index in indexs:
        # 过滤出当前 epoch 的数据
        x_current = averaged_data[:, index]
        y_current = 100-averaged_data[:, 22]
        x.append(x_current)
        y.append(y_current)


# Setting the figure size to match the uploaded image's aspect ratio
plt.figure(figsize=(10, 8))

# Plotting the data
plt.scatter(x[0], y[0], label='No Mixup', color='blue', marker='o')
plt.scatter(x[1], y[1], label='Mixup', color='orange', marker='d')

# Performing linear regression and plotting regression lines
colors = ['blue', 'orange']
for i in range(len(datasets)):
    coeffs = np.polyfit(x[i], y[i], 1)
    regression_line = np.poly1d(coeffs)
    x_range = np.linspace(min(x[i]), max(x[i]), 100)
    plt.plot(x_range, regression_line(x_range), label=None, linestyle='--', linewidth=3.5, color=colors[i])

# Adjusting font sizes to match the uploaded image as closely as possible
plt.title('Impact of Mixup', fontsize=28)
plt.xlabel('Region Counts', fontsize=28)
plt.ylabel('Generalization Gap', fontsize=28)
plt.legend(fontsize=24)
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=28)
plt.subplots_adjust(left=0.22, right=0.85, top=0.85, bottom=0.15)
root = 'mixup.png'
plt.savefig(root)  # 保存图像