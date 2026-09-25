import pandas as pd
import numpy as np

# ===================== 1. 原始数据录入 (完全匹配你的LaTeX表格 纯平均分数据) =====================
# 轮次
turns = [0, 1, 3, 5, 10]
# 模型列表
models = ['Qwen3-4B', 'Llama-3.1-8B', 'GPT-5-nano']

# 【Dual-Agent组】Baseline 与 缓解组 平均分
dual_baseline = {
    'Qwen3-4B': [0.96, 1.23, 1.80, 1.81, 1.21],
    'Llama-3.1-8B': [1.21, 1.36, 1.50, 1.04, 1.13],
    'GPT-5-nano': [0.86, 0.91, 0.98, 0.86, 1.17]
}
dual_mitigation = {
    'Qwen3-4B': [0.93, 1.19, 1.78, 2.20, 1.46],
    'Llama-3.1-8B': [0.82, 1.08, 2.45, 2.08, 1.63],
    'GPT-5-nano': [0.62, 0.64, 1.01, 1.04, 1.01]
}

# 【Multi-Agent组】Baseline 与 缓解组 平均分
multi_baseline = {
    'Qwen3-4B': [0.93, 1.75, 1.97, 1.46, 1.42],
    'Llama-3.1-8B': [1.12, 1.76, 1.65, 1.40, 1.43],
    'GPT-5-nano': [0.95, 0.97, 1.14, 0.97, 0.84]
}
multi_mitigation = {
    'Qwen3-4B': [0.71, 2.00, 2.05, 2.01, 1.60],
    'Llama-3.1-8B': [0.74, 1.59, 1.52, 1.26, 1.31],
    'GPT-5-nano': [0.73, 0.73, 0.93, 0.94, 0.94]
}

# ===================== 2. 核心计算：计算缓解效果差值 =====================
def calculate_diff(baseline_dict, mitigation_dict):
    """计算 基线值 - 缓解值 = 缓解效果差值"""
    diff_result = {}
    for model in models:
        diff_result[model] = np.round(np.array(baseline_dict[model]) - np.array(mitigation_dict[model]), 2)
    return diff_result

# 计算两组差值
dual_diff = calculate_diff(dual_baseline, dual_mitigation)
multi_diff = calculate_diff(multi_baseline, multi_mitigation)

# ===================== 3. 构建统计表格 =====================
table_data = []
for model in models:
    row = {
        'Model': model,
        'Dual-T0': dual_diff[model][0],
        'Dual-T1': dual_diff[model][1],
        'Dual-T3': dual_diff[model][2],
        'Dual-T5': dual_diff[model][3],
        'Dual-T10': dual_diff[model][4],
        'Multi-T0': multi_diff[model][0],
        'Multi-T1': multi_diff[model][1],
        'Multi-T3': multi_diff[model][2],
        'Multi-T5': multi_diff[model][3],
        'Multi-T10': multi_diff[model][4]
    }
    table_data.append(row)

# 转为DataFrame，方便格式化和导出
df = pd.DataFrame(table_data)

# ===================== 4. 增加【关键统计列】- 便于论文分析 =====================
# 各模型在 Dual/Multi 下的 平均缓解效果、最大缓解效果
df['Dual-Agent-平均缓解效果'] = np.round(df[['Dual-T0','Dual-T1','Dual-T3','Dual-T5','Dual-T10']].mean(axis=1), 2)
df['Multi-Agent-平均缓解效果'] = np.round(df[['Multi-T0','Multi-T1','Multi-T3','Multi-T5','Multi-T10']].mean(axis=1), 2)
df['Dual-Agent-最大缓解效果'] = df[['Dual-T0','Dual-T1','Dual-T3','Dual-T5','Dual-T10']].max(axis=1)
df['Multi-Agent-最大缓解效果'] = df[['Multi-T0','Multi-T1','Multi-T3','Multi-T5','Multi-T10']].max(axis=1)

# ===================== 5. 输出结果：①控制台打印美观表格 ②导出Excel文件 =====================
print("="*90)
print("📊 Prompt-Based Bias Mitigation Effect Analysis (差值 = Baseline - Mitigation)")
print("✅ 正值 = 有效缓解 | ❌ 负值 = 无缓解/偏见加剧 | 值越大 = 缓解效果越好")
print("="*90)
print(df.to_string(index=False))

# 导出Excel文件（可直接插入论文，完美适配）
excel_name = "bias_mitigation_effect_analysis.xlsx"
df.to_excel(excel_name, index=False, sheet_name="缓解效果差值统计")
print(f"\n✅ 统计表格已导出至: {excel_name}")
print("="*90)