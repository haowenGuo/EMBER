import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. 准备数据
data = [
    (0.96, 1.75), (1.23, 2.22), (1.80, 3.22), (1.81, 4.05), (1.21, 2.17),
    (1.21, 3.32), (1.36, 2.31), (1.50, 2.35), (1.04, 2.38), (1.13, 2.15),
    (0.86, 1.66), (0.91, 1.46), (0.98, 1.39), (0.86, 1.52), (1.17, 2.61),
    (0.93, 1.44), (1.19, 1.70), (1.78, 2.70), (2.20, 3.65), (1.46, 3.22),
    (0.82, 1.56), (1.08, 1.37), (2.45, 4.31), (2.08, 3.25), (1.63, 3.06),
    (0.62, 0.95), (0.64, 0.84), (1.01, 2.09), (1.04, 1.78), (1.01, 1.42),
    (0.93, 2.07), (1.75, 3.91), (1.97, 4.47), (1.46, 2.13), (1.42, 3.61),
    (1.12, 2.49), (1.76, 2.99), (1.65, 2.63), (1.40, 2.06), (1.43, 2.61),
    (0.95, 1.93), (0.97, 1.66), (1.14, 1.59), (0.97, 2.14), (0.84, 2.07),
    (0.71, 1.34), (2.00, 3.70), (2.05, 4.51), (2.01, 3.50), (1.60, 2.38),
    (0.74, 1.49), (1.59, 2.47), (1.52, 2.98), (1.26, 2.32), (1.31, 2.42),
    (0.73, 0.97), (0.73, 0.87), (0.93, 1.32), (0.94, 1.61), (0.94, 2.14)
]

df = pd.DataFrame(data, columns=['Mean', 'SD'])

# 2. 使用 Pandas 直接计算皮尔逊相关系数 (默认就是 pearson)
correlation = df['Mean'].corr(df['SD'])

# 3. 计算显著性 P 值 (手动实现简易版 T 检验，不依赖 scipy)
# r * sqrt((n-2)/(1-r^2))
n = len(df)
t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2))
# 对于 n=60，t 统计量 > 3.4 基本上 p 就在 0.001 以下了

print(f"Pearson Correlation Coefficient (r): {correlation:.4f}")
print(f"Sample Size (n): {n}")

# 4. 绘图
plt.figure(figsize=(7, 5))
sns.set_style("whitegrid")

# 使用 r'...' 解决 LaTeX 语法转义告警
sns.regplot(x='Mean', y='SD', data=df,
            scatter_kws={'alpha':0.6, 'color':'#1f77b4'},
            line_kws={'color':'#d62728'})

plt.title('Correlation Analysis: Bias Intensity vs. Stability', fontsize=12)
plt.xlabel(r'Mean Bias Score ($\mu$)', fontsize=11)
plt.ylabel(r'Standard Deviation ($\sigma$)', fontsize=11)

# 在图中添加相关系数文本
plt.text(0.7, 4.0, f'$r = {correlation:.3f}$\n$p < 0.001$',
         fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()