import matplotlib.pyplot as plt
import numpy as np

# ---------------------- 1. 全局数据整理（已替换为你的提示词数据） ----------------------
# 只保留你表格中的轮次：0, 1, 3, 5, 10
turns = np.array([0, 1, 3, 5, 10])
turn_labels = [f'Turn {i}' for i in turns]

# ---------------------- 你的表格数据 ----------------------
# 均值数据 (Mean)
neutral_mean = np.array([1.11, 0.98, 0.97, 0.73, 0.48])    # Neutral Prompt
mild_mean = np.array([0.98, 0.81, 0.69, 0.70, 0.29])       # Mild Prompt
adversarial_mean = np.array([0.96, 1.23, 1.80, 1.81, 1.21]) # Adversarial Prompt

# 方差数据 (Variance, σ²)
neutral_var = np.array([2.06, 1.60, 1.73, 1.65, 0.84])
mild_var = np.array([1.53, 1.19, 1.08, 1.32, 0.45])
adversarial_var = np.array([1.75, 2.22, 3.22, 4.05, 2.17])

# 将数据整理为数组格式，方便绘图
all_means = np.array([neutral_mean, mild_mean, adversarial_mean])
all_vars = np.array([neutral_var, mild_var, adversarial_var])

# ---------------------- 2. 绘图全局设置 ----------------------
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10  # 稍微调大字体，适配你的数据
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.dpi'] = 300

# 配色：对应三种提示词
distinct_colors = ['#1f77b4', '#ff4b5c', '#000000'] # 蓝/红/黑
markers = ['o', 's', '*']
scenario_labels = [
    'Neutral Prompt',
    'Mild Prompt',
    'Adversarial Prompt'
]

# ---------------------- 3. 绘图函数（已适配你的数据） ----------------------
def plot_prompt_evolution(means, vars, plot_variance=False):
    """
    绘制提示词强度演化图
    means: 三种提示词的均值数组
    vars: 三种提示词的方差数组
    plot_variance: 是否绘制方差图（默认只绘制均值）
    """
    # --- 1. 绘制均值图 ---
    fig, ax = plt.subplots(figsize=(8, 4))

    for i in range(3):
        ax.plot(turns, means[i],
                color=distinct_colors[i], marker=markers[i], linestyle='-',
                linewidth=1.8, markersize=7, label=scenario_labels[i])

    # 图表美化
    ax.set_xlabel('Debate Rounds (Turn)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Prompt Strength (Mean, $\mu$)', fontsize=12, fontweight='bold')
    ax.set_title('Bias Intensity Evolution under Different Prompts', fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks(turns)
    ax.set_xticklabels(turn_labels, fontsize=10)
    ax.set_ylim(bottom=0, top=2.2) # 适配你的数据范围
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(frameon=True, loc='upper right', ncol=1, fontsize=10)

    plt.tight_layout()
    plt.savefig('Prompt_Strength_Mean.png', format='png', bbox_inches='tight')
    plt.savefig('Prompt_Strength_Mean.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    # --- 2. 绘制方差图（可选） ---
    if plot_variance:
        fig, ax = plt.subplots(figsize=(8, 4))

        for i in range(3):
            ax.plot(turns, vars[i],
                    color=distinct_colors[i], marker=markers[i], linestyle='-',
                    linewidth=1.8, markersize=7, label=scenario_labels[i])

        ax.set_xlabel('Debate Rounds (Turn)', fontsize=4, fontweight='bold')
        ax.set_ylabel('Prompt Strength (Variance, $\sigma^2$)', fontsize=4, fontweight='bold')
        ax.set_title('Bias Polarization under Different Prompts', fontsize=4, fontweight='bold', pad=10)
        ax.set_xticks(turns)
        ax.set_xticklabels(turn_labels, fontsize=6)
        ax.set_ylim(bottom=0, top=4.5) # 适配你的方差范围
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.legend(frameon=True, loc='upper right', ncol=1, fontsize=6)

        plt.tight_layout()
        plt.savefig('Prompt_Strength_Variance.png', format='png', bbox_inches='tight')
        plt.savefig('Prompt_Strength_Variance.pdf', format='pdf', bbox_inches='tight')
        plt.show()

# ---------------------- 4. 生成图片 ----------------------
# 执行绘图（plot_variance=True 表示同时生成方差图）
plot_prompt_evolution(all_means, all_vars, plot_variance=True)