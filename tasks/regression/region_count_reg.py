import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 读取数据
root = 'all.txt'
data = np.genfromtxt(root)
print("原始数据的行数:", data.shape[0])

# 数据预处理
averaged_data = data
print("处理后数据的行数:", averaged_data.shape[0])

# 提取自变量和因变量
X = averaged_data[:, [4, 5]]  # 自变量: [1d, 2d, d(W), γ(W), γ(W)/d(W)]
y = averaged_data[:, 2] 

# 添加截距项（用于statsmodels）
X_with_const = sm.add_constant(X)

# ==================== 1. 多元回归分析（系数 + 显著性） ====================
model = sm.OLS(y, X_with_const).fit()
print("\n=== 多元回归分析结果 ===")
print(model.summary())  # 输出完整回归报告（含系数、p值、R²等）

# ==================== 2. 计算Partial R² ====================
def partial_r2(model, X, target_var_idx):
    full_r2 = model.rsquared
    reduced_X = np.delete(X, target_var_idx, axis=1)
    reduced_model = sm.OLS(y, reduced_X).fit()
    return full_r2 - reduced_model.rsquared

partial_r2s = [partial_r2(model, X_with_const, i) for i in range(1, X_with_const.shape[1])]  # 跳过截距项
