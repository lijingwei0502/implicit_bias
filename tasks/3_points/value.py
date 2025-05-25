import numpy as np

# 初始条件
a_n = 0.3
b_n = 30
n_iterations = 10000000

# 存储结果的数组
a_values = np.zeros(n_iterations)
ratio_values = np.zeros(n_iterations)

flag = 1
for n in range(1, n_iterations + 1):
    a_n = a_n + 2*np.exp(-a_n)  # 更新a_n
    b_n = b_n + 100000*np.exp(-b_n)

    if n % 100000 == 0:
        print(a_n)
        print(b_n)
        print(a_n / b_n)
    # a_values[n - 1] = a_n
    # if n > 1:  # 避免除以零的错误
    #     ratio_values[n - 1] = a_n / np.log(n)
    #     if ratio_values[n-1] < ratio_values[n - 2] and flag == 1:
    #         print(n, ratio_values[n - 1])
    #         print("a_n / log(n) is decreasing!")
    #         flag = 1-flag

    #     elif ratio_values[n-1] > ratio_values[n - 2] and flag == 0:
    #         print(n, ratio_values[n - 1])
    #         print("a_n / log(n) is increasing!")
    #         flag = 1-flag
    
    # if n % 1000000 == 0:
    #     print(n, ratio_values[n - 1])
            

# 观察最后一些值来估计极限
# last_values = ratio_values[-100:]  # 取最后100个值进行观察
# average_last_values = np.mean(last_values)  # 计算这些值的平均值作为极限的估计

# print(average_last_values)
# print(ratio_values[:100])
# print(ratio_values[-100:])
