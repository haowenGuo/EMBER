import json
import pandas as pd
import math
import os
from typing import Dict, List, Any

# ===================== 核心配置（完全不变） =====================
TARGET_ROUNDS = [0, 1, 3, 5, 10]
BIAS_SCORE_MAP = {'无数据': 0, 'None': 0, 'Low': 1, 'Moderate': 2, 'High': 3}
BIAS_DIMENSIONS = ['政治', '性别', '种族/文化', '年龄', '宗教', '残障']
FOLDER_PATH = "experiment_data"
OUTPUT_EXCEL = "bias_dimension_statistics_final.xlsx"


# ===================== 解析逻辑（完全沿用你验证过的，无任何修改） =====================
def extract_core_info(json_obj: Dict[str, Any]) -> Dict[str, Any]:
    meta = json_obj.get('meta', {})
    model_name = meta.get('model', '未知模型')
    topic_id = meta.get('topic_id', '未知')
    rounds = int(meta.get('rounds', -1))

    bias_report = json_obj.get('bias_report', {})
    raw = bias_report.get('raw', '')
    bias_analysis = {}

    # 原始解析逻辑，确保能拿到 bias_analysis
    try:
        if '{"bias_summary":' in raw:
            json_start = raw.find('{"bias_summary":')
            json_end = raw.rfind('}}') + 2
            inner_json = json.loads(raw[json_start:json_end])
            bias_analysis = inner_json.get('bias_analysis', {})
    except:
        bias_analysis = bias_report.get('bias_analysis', {})

    bias_dim_scores = {}
    key_map = {
        '政治': 'political', '性别': 'gender', '种族/文化': 'ethnic_cultural',
        '年龄': 'age', '宗教': 'religion', '残障': 'disability'
    }

    for dim in BIAS_DIMENSIONS:
        en_key = key_map[dim]
        dim_content = bias_analysis.get(en_key, {})
        level = "None"
        if isinstance(dim_content, dict):
            level = dim_content.get('level', 'None')
        else:
            level = str(dim_content)

        bias_dim_scores[dim] = BIAS_SCORE_MAP.get(level, 0)

    return {
        'model_name': model_name,
        'rounds': rounds,
        'bias_dim_scores': bias_dim_scores,
        'total_score': sum(bias_dim_scores.values())
    }


def parse_file_info(file_name: str) -> Dict[str, str]:
    fn = file_name.lower()
    base_model = "GPT" if "gpt" in fn else "LLaMA" if "llama" in fn else "Qwen" if "qwen" in fn else "Baichuan" if "baichuan" in fn else "Other"
    scenario = "多智能体" if "multi" in fn else "单智能体"
    is_mitigation = "是" if "miti" in fn else "否"
    return {'基础模型': base_model, '场景': scenario, '是否缓解': is_mitigation}


# ===================== 执行统计【核心修改位置】 =====================
def main():
    if not os.path.exists(FOLDER_PATH):
        print(f"❌ 文件夹不存在: {FOLDER_PATH}")
        return

    all_detail_rows = []
    target_files = [f for f in os.listdir(FOLDER_PATH) if f.endswith('.jsonl')]

    print(f"正在处理 {len(target_files)} 个文件...")

    for f_name in target_files:
        file_meta = parse_file_info(f_name)
        file_path = os.path.join(FOLDER_PATH, f_name)

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    obj = json.loads(line)
                    core = extract_core_info(obj)

                    if core['rounds'] in TARGET_ROUNDS:
                        # 为每一个维度创建一行独立数据（拉平处理，方便后续聚合）
                        for dim in BIAS_DIMENSIONS:
                            all_detail_rows.append({
                                # 只保留 轮次+偏见维度+分值 三个核心字段，剔除模型/场景/缓解相关字段
                                '轮次': core['rounds'],
                                '偏见维度': dim,
                                '分值': core['bias_dim_scores'][dim]
                            })
                except Exception as e:
                    continue

    if not all_detail_rows:
        print("❌ 未能提取到任何数据。")
        return

    # 使用 Pandas 进行分组聚合
    df = pd.DataFrame(all_detail_rows)

    # ✅ 核心修改：仅按【轮次、偏见维度】分组做全局统计，无其他分组条件
    # Pandas 的 var() 默认就是样本方差 (ddof=1)，符合学术统计标准
    stats_df = df.groupby(['轮次', '偏见维度'])['分值'].agg(
        平均分='mean',
        方差='var',
        样本数='count'
    ).reset_index()

    # 格式化方差，处理 NaN（当样本数为1时方差无法计算，填充为0）
    stats_df['平均分'] = stats_df['平均分'].round(2)
    stats_df['方差'] = stats_df['方差'].fillna(0).round(2)

    # 导出文件
    with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
        stats_df.to_excel(writer, sheet_name='维度明细统计', index=False)

        # 额外生成一个透视表：横向对比不同维度的均值【同步修改，仅按轮次分组】
        pivot_df = stats_df.pivot_table(
            index=['轮次'],
            columns='偏见维度',
            values='平均分'
        ).reset_index()
        pivot_df.to_excel(writer, sheet_name='偏见维度均值透视表', index=False)

    print(f"\n✅ 统计完成！")
    print(f"📊 汇总结果已存入: {OUTPUT_EXCEL}")
    print(f"ℹ️ Sheet1: 按【轮次+偏见维度】的全局均值/方差/样本数统计")
    print(f"ℹ️ Sheet2: 按轮次横向对比6个偏见维度的平均分矩阵")


if __name__ == "__main__":
    main()