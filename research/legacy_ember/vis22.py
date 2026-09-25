import matplotlib.pyplot as plt
import numpy as np

# 1. 数据准备
# 对话轮次 (Turn 0 - Turn 10)
turns = np.arange(0, 11)

# 三条曲线的均值数据 (μ)
baseline = [0.62, 0.89, 0.83, 0.78, 1.04, 0.98, 0.95, 1.04, 1.02, 1.06, 1.18]
prompt_mit = [0.80, 0.88, 0.95, 1.02, 1.04, 1.02, 1.01, 1.00, 1.01, 1.01, 1.02]
ember_agent = [0.75, 0.83, 0.76, 0.85, 0.98, 0.97, 1.10, 1.35, 1.15, 0.93, 1.17]

# 2. 绘图设置
plt.figure(figsize=(10, 6))

# 绘制三条折线，严格匹配样式
plt.plot(turns, baseline, marker='o', color='#1f77b4', linewidth=2.5, markersize=6, label='Baseline (Dual-Agent)')
plt.plot(turns, prompt_mit, marker='s', color='#ff7f0e', linewidth=2.5, markersize=6, label='Prompt Mitigation (Dual-Agent)')
plt.plot(turns, ember_agent, marker='*', color='black', linewidth=2.5, markersize=8, label='EMBER-Agent')

# 3. 图表样式（严格匹配原图）
plt.title('GPT: Bias Evolution', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Dialogue Turn', fontsize=14, labelpad=10)
plt.ylabel('Bias Score Mean ($\\mu$)', fontsize=14, labelpad=10)
plt.xticks(turns, [f'Turn {i}' for i in turns], fontsize=12)
plt.yticks(np.arange(0.0, 1.7, 0.2), fontsize=12)
plt.ylim(0.0, 1.6)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper left', fontsize=12)
plt.tight_layout()

# 4. 显示图表
plt.show()