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
X = averaged_data[:, [0, 2, 4, 5]]  # 自变量: [1d, 2d, d(W), γ(W), γ(W)/d(W)]
y = averaged_data[:, 1] - averaged_data[:, 7]  # 因变量: Generalization Gap

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

# ==================== 3. 计算VIF（检查共线性） ====================
vifs = [variance_inflation_factor(X_with_const, i) for i in range(1, X_with_const.shape[1])]  # 跳过截距项

# ==================== 4. 结果可视化 ====================
# (1) 绘制回归系数和显著性
coefs = model.params[1:]  # 忽略截距项
p_values = model.pvalues[1:]
variables = ['1d', '2d', 'd(W)', 'γ(W)']

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
bars = plt.bar(variables, coefs, color=np.where(p_values < 0.05, 'blue', 'gray'))
plt.axhline(0, color='black', linestyle='--')
plt.title('回归系数 (蓝色=显著)', fontsize=12)
plt.ylabel('系数值')

# (2) 绘制Partial R²
plt.subplot(1, 2, 2)
plt.bar(variables, partial_r2s, color='green')
plt.title('Partial R²', fontsize=12)
plt.ylabel('解释力')

plt.tight_layout()
plt.savefig("regression_analysis.png")
plt.show()

# ==================== 5. 打印关键结果 ====================
print("\n=== 关键指标 ===")
print(f"多元回归 R²: {model.rsquared:.4f}")
print("\nPartial R²:")
for var, pr2 in zip(variables, partial_r2s):
    print(f"  - {var}: {pr2:.4f}")

print("\nVIF (检查共线性):")
for var, vif in zip(variables, vifs):
    print(f"  - {var}: {vif:.2f}")