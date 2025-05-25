import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

plt.rcParams['figure.dpi'] = 300

datasets = ['primal', 'augmentation']
all_x = []
all_y = []

for dataset in datasets:
    root = str(dataset) + '.txt'
    data = np.genfromtxt(root)

    print("原始数据的行数:", data.shape[0])

    dtype = [('col' + str(i), float) for i in range(data.shape[1])]
    structured_data = np.core.records.fromarrays(data.T, dtype=dtype)

    # 排序和分组
    sorted_data = np.sort(structured_data, order=['col23', 'col24', 'col25'])
    unique_keys, indices = np.unique(sorted_data[['col23', 'col24', 'col25']], return_index=True, axis=0)
    grouped_data = np.split(sorted_data, indices[1:])

    averaged_data = []
    for group in grouped_data:
        group_array = np.array(group.tolist())
        averaged_data.append(group_array.mean(axis=0))
    averaged_data = np.array(averaged_data)

    print("处理后数据的行数:", averaged_data.shape[0])

    # index = 20 对应 Region Count
    x_current = averaged_data[:, 20]
    y_current = averaged_data[:, 21] - averaged_data[:, 22]
    
    all_x.append(x_current)
    all_y.append(y_current)

# 合并所有数据用于统一回归分析
x_total = np.concatenate(all_x)
y_total = np.concatenate(all_y)

# 计算 Pearson 相关系数
corr, p_value = pearsonr(x_total, y_total)

# 绘图
plt.figure(figsize=(10, 8))
plt.scatter(x_total, y_total, color='purple', alpha=0.6, label='All Data')

# 拟合并画出整体回归线
coeffs = np.polyfit(x_total, y_total, 1)
regression_line = np.poly1d(coeffs)
x_range = np.linspace(min(x_total), max(x_total), 100)
plt.plot(x_range, regression_line(x_range), linestyle='--', linewidth=3.5, color='black', label='Regression Line')

# 标注相关性值
plt.title(f'Correlation: {corr:.2f}', fontsize=28)
plt.xlabel('Region Counts', fontsize=28)
plt.ylabel('Generalization Gap', fontsize=28)
plt.legend(fontsize=24)
plt.tick_params(axis='both', which='major', labelsize=24)
plt.grid(True)
plt.subplots_adjust(left=0.22, right=0.85, top=0.85, bottom=0.15)

plt.savefig('correlation_all.png')
