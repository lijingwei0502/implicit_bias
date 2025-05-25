import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams['figure.dpi'] = 300

batchs = [256,1024]

x = []
y = []
for batch in batchs:
    root = 'Resnet18_' + str(batch) + '.txt'
    # 如果路径不存在，则跳过
    if not os.path.exists(root):
        continue
    data = np.genfromtxt(root)

    # 打印原始数据的行数
    print("原始数据的行数:", data.shape[0])

    averaged_data = data
    
    filtered_data = averaged_data[averaged_data[:, 2] > 1e-07]

    #filtered_data = filtered_data[filtered_data[:, 1] <= 0.05]
    # 打印处理后数据的行数
    print("处理后数据的行数:", filtered_data.shape[0])

    # x_current 和 y_current 现在是过滤后的数据
    x_current = filtered_data[:, 0]
    y_current = filtered_data[:, 1]/batch
    x.append(x_current)
    y.append(y_current)
    
# Plotting the data
plt.figure(figsize=(8.7, 7))
plt.scatter(x[0], y[0], label='Minibatch', color='blue', marker='o')
plt.scatter(x[1], y[1], label='Largebatch', color='red', marker='v')
plt.legend(fontsize=16)
plt.title('Average Regions vs. Noise', fontsize=18)
plt.xlabel('Average Regions',fontsize=18)
#plt.ylabel('Generalization Gap')
plt.ylabel('Noise',fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=18)
# 调整边距
plt.subplots_adjust(left=0.2, right=0.85, top=0.85, bottom=0.15)  # 这里的数值可以根据需要调整
plt.savefig('batch.png')  # 保存图像