import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import spearmanr  # Import Spearman's rank correlation

plt.rcParams['figure.dpi'] = 300

nets = ['Resnet18','Resnet34', 'VGG19', 'MobileNet', 'SENet18', 'ShuffleNetV2', 'EfficientNetB0', 'RegNetX_200MF', 'SimpleDLA']

for net in nets:
    root = str(net) + '.txt'
    if not os.path.exists(root):
        continue
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

    indexs = [20]
    plt.figure(figsize=(10, 8))
    for index in indexs:
        x_current = averaged_data[:, index]
        y_current = averaged_data[:, 21] - averaged_data[:, 22]
        
        # Calculate Spearman's rank correlation (and p-value)
        correlation, p_value = spearmanr(x_current, y_current)

        plt.scatter(x_current, y_current)
        plt.xlabel('Average Regions', fontsize=22)
        plt.ylabel('Test Accuracy', fontsize=22)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.title(f'Spearman Correlation of {net}: {correlation:.2f} (p={p_value:.3f})', fontsize=22)
        
        root = str(net) + '.png'
        plt.savefig(root)
        plt.clf()
        
        with open('correlation.txt', 'a') as f:
            f.write(f"{net} Spearman: {correlation:.4f} (p-value: {p_value:.4f})\n")