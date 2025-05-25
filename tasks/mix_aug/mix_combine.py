import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.dpi'] = 300

datasets = ['primal', 'mixup']
all_x = []
all_y = []

for dataset in datasets:
    root = str(dataset) + '.txt'
    data = np.genfromtxt(root)

    print("原始数据的行数:", data.shape[0])

    dtype = [('col' + str(i), float) for i in range(data.shape[1])]
    structured_data = np.core.records.fromarrays(data.T, dtype=dtype)

    sorted_data = np.sort(structured_data, order=['col23', 'col24', 'col25'])
    unique_keys, indices = np.unique(sorted_data[['col23', 'col24', 'col25']], return_index=True, axis=0)
    grouped_data = np.split(sorted_data, indices[1:])

    averaged_data = []
    for group in grouped_data:
        group_array = np.array(group.tolist())
        averaged_data.append(group_array.mean(axis=0))
    averaged_data = np.array(averaged_data)

    print("处理后数据的行数:", averaged_data.shape[0])

    # Region Count at index 20, generalization gap is 100 - test acc (col22)
    x_current = averaged_data[:, 20]
    y_current = 100 - averaged_data[:, 22]
    
    all_x.append(x_current)
    all_y.append(y_current)

# 合并所有数据
x_total = np.concatenate(all_x)
y_total = np.concatenate(all_y)

# 手动计算 Pearson correlation
x_mean = np.mean(x_total)
y_mean = np.mean(y_total)
numerator = np.sum((x_total - x_mean) * (y_total - y_mean))
denominator = np.sqrt(np.sum((x_total - x_mean)**2)) * np.sqrt(np.sum((y_total - y_mean)**2))
corr = numerator / denominator

# 绘图
plt.figure(figsize=(10, 8))
plt.scatter(x_total, y_total, color='purple', alpha=0.6, label='All Data')

# 拟合整体回归线
coeffs = np.polyfit(x_total, y_total, 1)
regression_line = np.poly1d(coeffs)
x_range = np.linspace(min(x_total), max(x_total), 100)
plt.plot(x_range, regression_line(x_range), linestyle='--', linewidth=3.5, color='black', label='Regression Line')

# 设置图形信息
# title report correlation
plt.title(f'Correlation: {corr:.2f}', fontsize=28)
plt.xlabel('Region Counts', fontsize=28)
plt.ylabel('Generalization Gap', fontsize=28)
plt.legend(fontsize=24)
plt.tick_params(axis='both', which='major', labelsize=24)
plt.grid(True)
plt.subplots_adjust(left=0.22, right=0.85, top=0.85, bottom=0.15)


plt.savefig('mixup_correlation.png')
