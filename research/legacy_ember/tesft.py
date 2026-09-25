import matplotlib.pyplot as plt
import numpy as np

# ---------------------- 1. 全局数据整理（保持不变） ----------------------
turns = np.arange(0, 11)
turn_labels = [f'Turn {i}' for i in turns]

# Qwen3-4B
qwen3_mean = np.array([
    [0.96, 1.23, 1.62, 1.80, 1.73, 1.81, 1.59, 1.47, 1.35, 1.26, 1.21],
    [0.93, 1.19, 1.57, 1.78, 2.06, 2.20, 1.98, 1.83, 1.70, 1.55, 1.46]
])
qwen3_var = np.array([
    [1.75, 2.22, 2.91, 3.22, 3.82, 4.05, 3.37, 2.98, 2.49, 2.28, 2.17],
    [1.44, 1.70, 2.38, 2.70, 3.32, 3.65, 3.42, 3.21, 3.35, 3.19, 3.22]
])
qwen3_std = np.sqrt(qwen3_var)

# Llama-3.1-8B
llama3_mean = np.array([
    [1.21, 1.36, 1.49, 1.50, 1.21, 1.04, 1.01, 1.07, 1.12, 1.11, 1.13],
    [0.82, 1.08, 1.89, 2.45, 2.32, 2.08, 1.92, 1.81, 1.73, 1.67, 1.63]
])
llama3_var = np.array([
    [3.32, 2.31, 2.48, 2.35, 2.49, 2.38, 2.21, 2.19, 2.26, 2.18, 2.15],
    [1.56, 1.37, 2.97, 4.31, 3.61, 3.25, 3.08, 3.22, 3.11, 3.09, 3.06]
])
llama3_std = np.sqrt(llama3_var)

# GPT-5-nano
gpt5_mean = np.array([
    [0.86, 0.91, 0.97, 0.98, 0.89, 0.86, 0.93, 1.00, 1.06, 1.10, 1.17],
    [0.62, 0.64, 0.87, 1.01, 1.03, 1.04, 1.02, 1.01, 1.00, 1.01, 1.01]
])
gpt5_var = np.array([
    [1.66, 1.46, 1.53, 1.39, 1.51, 1.52, 1.82, 1.97, 2.15, 2.41, 2.61],
    [0.95, 0.84, 1.58, 2.09, 1.87, 1.78, 1.62, 1.59, 1.52, 1.47, 1.42]
])
gpt5_std = np.sqrt(gpt5_var)

# EMBER-Agent模型
debater_qwen_mean = np.array([0.71, 1.17, 1.54, 1.42, 1.49, 1.17, 1.26, 1.39, 1.15, 1.25, 1.15])
debater_qwen_std = np.sqrt(np.array([1.22, 1.92, 3.24, 2.81, 2.21, 1.86, 1.79, 1.84, 1.50, 2.05, 1.74]))

debater_llama_mean = np.array([0.90, 0.94, 1.07, 1.11, 0.92, 0.93, 0.88, 0.77, 0.92, 0.73, 0.85])
debater_llama_std = np.sqrt(np.array([1.75, 1.30, 1.90, 1.85, 1.50, 1.57, 1.89, 0.78, 1.39, 1.42, 1.21]))

#debater_gpt_mean = np.array([0.75, 0.83, 0.76, 0.95, 0.98, 1.04, 0.97, 1.10, 1.34, 1.14, 0.93])
#debater_gpt_std = np.sqrt(np.array([1.01, 0.84, 0.98, 2.00, 1.39, 1.56, 1.09, 1.23, 2.06, 1.99, 1.34]))
debater_gpt_mean = np.array([0.62, 0.89, 0.83, 0.78, 1.04, 0.98, 0.95, 0.80, 1.04, 1.02, 1.06])
debater_gpt_std = np.sqrt(np.array([0.8, 1.39, 1.15, 1.1, 1.11, 1.33, 1.62, 1.13, 1.13, 1.25, 1.61]))

# ---------------------- 2. 绘图全局设置 ----------------------
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 6
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.dpi'] = 300

# 配色保持不变
distinct_colors = {
    'scenarios': ['#1f77b4', '#ff4b5c'],
    'debater': '#000000'
}
markers = ['o', 's', '*']
scenario_labels = [
    'Baseline (Dual-Agent)',
    'Prompt Mitigation (Dual-Agent)',
    'EMBER-Agent'
]

# ---------------------- 3. 独立绘图函数（已移除误差线） ----------------------
def plot_single_model(means, debater_mean, model_name, ylim_max, filename_prefix):
    """绘制单个模型的均值演化图（窄宽度+无误差线）"""
    fig, ax = plt.subplots(figsize=(8, 4))

    # 绘制2个基准场景（纯趋势线，无误差棒）
    for i in range(2):
        ax.plot(turns, means[i],
                color=distinct_colors['scenarios'][i], marker=markers[i], linestyle='-',
                linewidth=1.4, markersize=5.5, label=scenario_labels[i])

    # 绘制EMBER-Agent模型
    ax.plot(turns, debater_mean,
            color=distinct_colors['debater'], marker=markers[2], linestyle='--',
            linewidth=1.7, markersize=6.5, label=scenario_labels[2])

    # 图表美化
    ax.set_xlabel('Dialogue Turn', fontsize=8, fontweight='bold')
    ax.set_ylabel('Bias Score Mean ($\mu$)', fontsize=8, fontweight='bold')
    ax.set_title(f'{model_name}: Bias Evolution', fontsize=8, fontweight='bold', pad=10)
    ax.set_xticks(turns)
    ax.set_xticklabels(turn_labels, fontsize=6)
    ax.set_ylim(bottom=0, top=ylim_max)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(frameon=True, loc='upper right', ncol=1, fontsize=8.5)

    # 锁定布局，保证Y轴标签完整
    plt.tight_layout(rect=[0.14, 0, 0.99, 0.98])
    plt.savefig(f'{filename_prefix}_mean_clean.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f'{filename_prefix}_mean_clean.png', format='png', bbox_inches='tight')
    plt.show()


def plot_single_model_variance(vars, debater_var, model_name, ylim_max, filename_prefix):
    """绘制单个模型的方差演化图（保持不变）"""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    for i in range(2):
        ax.plot(turns, vars[i], color=distinct_colors['scenarios'][i], marker=markers[i],
                linestyle='-', linewidth=1.4, markersize=5.5, label=scenario_labels[i])

    ax.plot(turns, debater_var, color=distinct_colors['debater'], marker=markers[2],
            linestyle='--', linewidth=1.7, markersize=6.5, label=scenario_labels[2])

    ax.set_xlabel('Dialogue Turn', fontsize=11, fontweight='bold')
    ax.set_ylabel('Bias Score Variance ($\sigma^2$)', fontsize=11, fontweight='bold')
    ax.set_title(f'{model_name}: Bias Polarization', fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks(turns)
    ax.set_xticklabels(turn_labels, fontsize=9)
    ax.set_ylim(bottom=0, top=ylim_max)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(frameon=True, loc='upper right', ncol=1, fontsize=8.5)

    plt.tight_layout(rect=[0.14, 0, 0.99, 0.98])
    plt.savefig(f'{filename_prefix}_variance_clean.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f'{filename_prefix}_variance_clean.png', format='png', bbox_inches='tight')
    plt.show()


# ---------------------- 4. 批量生成图片 ----------------------
# --- 1. Qwen系列 ---
plot_single_model(qwen3_mean, debater_qwen_mean,
                  model_name='Qwen', ylim_max=2.5, filename_prefix='Qwen_Final')


# --- 2. Llama系列 ---
plot_single_model(llama3_mean, debater_llama_mean,
                  model_name='Llama', ylim_max=2.8, filename_prefix='Llama_Final')


# --- 3. GPT系列 ---
plot_single_model(gpt5_mean, debater_gpt_mean,
                  model_name='GPT', ylim_max=1.6, filename_prefix='GPT_Final')
