import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams['figure.dpi'] = 300



root1 ='1d.txt'
root2 ='2d.txt'
root3 ='3d.txt'

data1 = np.genfromtxt(root1)
data2 = np.genfromtxt(root2)
data3 = np.genfromtxt(root3)

# 假设最后四列是相同的，我们取data1的最后四列作为代表
last_four_columns = data1[:, -4:]

# 拼接前面的列（除了最后四列）
combined_data = np.hstack((data1[:, :-4], data2[:, :-4], data3[:, :-4]))

# 将拼接后的数据与最后四列合并
final_data = np.hstack((combined_data, last_four_columns))

# 如果需要保存合并后的数据
np.savetxt('combined_data.txt', final_data, delimiter=' ', fmt='%g')

# 打印合并后的数据形状
print("Combined data shape:", final_data.shape)

