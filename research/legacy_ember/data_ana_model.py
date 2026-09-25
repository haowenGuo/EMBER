import json
import pandas as pd
import math
import os
from collections import defaultdict
from typing import Dict, List, Any

# ===================== 核心配置 =====================
TARGET_ROUNDS = [0, 1, 3, 5, 10]
BIAS_SCORE_MAP = {'无数据': 0, 'None': 0, 'Low': 1, 'Moderate': 2, 'High': 3}
BIAS_DIMENSIONS = ['政治', '性别', '种族/文化', '年龄', '宗教', '残障']
FOLDER_PATH = "experiment_data"
OUTPUT_EXCEL = "bias_all_model_complete_statistics_v6.xlsx"


# ===================== 工具函数（保留你的原始解析） =====================
def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    if not os.path.exists(file_path):
        return data
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line: continue
            try:
                data.append(json.loads(line))
            except Exception as e:
                print(f"⚠️ 解析失败: {e}")
    return data


def extract_core_info(json_obj: Dict[str, Any]) -> Dict[str, Any]:
    """完全保留你的原始解析逻辑"""
    meta = json_obj.get('meta', {})
    model_name = meta.get('model', '未知模型')
    topic_id = meta.get('topic_id', '未知')
    rounds = int(meta.get('rounds', -1))

    bias_report = json_obj.get('bias_report', {})
    raw = bias_report.get('raw', '')
    bias_analysis = {}
    try:
        if '{"bias_summary":' in raw:
            json_start = raw.find('{"bias_summary":')
            json_end = raw.rfind('}}') + 2
            inner_json = json.loads(raw[json_start:json_end])
            bias_analysis = inner_json.get('bias_analysis', {})
    except:
        bias_analysis = bias_report.get('bias_analysis', {})

    bias_levels = {}
    bias_dim_scores = {}
    key_map = {
        '政治': 'political', '性别': 'gender', '种族/文化': 'ethnic_cultural',
        '年龄': 'age', '宗教': 'religion', '残障': 'disability'
    }
    for dim in BIAS_DIMENSIONS:
        level = bias_analysis.get(key_map[dim], {}).get('level', 'None')
        bias_levels[dim] = level
        bias_dim_scores[dim] = BIAS_SCORE_MAP.get(level, 0)

    return {
        'model_name': model_name, 'topic_id': topic_id, 'rounds': rounds,
        'bias_levels': bias_levels, 'bias_dim_scores': bias_dim_scores,
        'total_score': sum(bias_dim_scores.values())
    }


def parse_file_info(file_name: str) -> Dict[str, str]:
    fn = file_name.lower()
    base_model = "GPT" if "gpt" in fn else "LLaMA" if "llama" in fn else "Qwen" if "qwen" in fn else "Baichuan" if "baichuan" in fn else "Unknown"
    scenario = "多智能体" if "multi" in fn else "单智能体"
    is_mitigation = "是" if "miti" in fn else "否"
    return {
        '基础模型': base_model, '场景': scenario, '是否缓解': is_mitigation,
        '完整标识': f"{base_model}-{scenario}-{'有缓解' if is_mitigation == '是' else '无缓解'}"
    }


def calculate_variance(score_list: List[float], avg_score: float) -> float:
    """计算样本方差"""
    if len(score_list) <= 1: return 0.00
    # 使用样本方差公式 (n-1)
    var = sum([(s - avg_score) ** 2 for s in score_list]) / (len(score_list) - 1)
    return round(var, 4)  # 提高精度到4位，方便后续观察


# ===================== 核心统计逻辑 =====================
def process_all_data():
    all_rows = []
    if not os.path.exists(FOLDER_PATH):
        print(f"❌ 找不到文件夹: {FOLDER_PATH}")
        return []

    target_files = [f for f in os.listdir(FOLDER_PATH) if f.endswith('.jsonl')]

    for file_name in target_files:
        file_info = parse_file_info(file_name)
        data = load_jsonl(os.path.join(FOLDER_PATH, file_name))

        for item in data:
            core = extract_core_info(item)
            if core['rounds'] in TARGET_ROUNDS:
                # 将文件信息和解析出的偏见分合并为一行明细
                row = {**file_info, **core}
                all_rows.append(row)

    return all_rows


def generate_reports(all_rows: List[Dict]):
    df = pd.DataFrame(all_rows)

    # --- 工作表1：模型-场景-轮次统计 (总分维度) ---
    # 修改：显式调用 calculate_variance 替代 std
    model_stat = df.groupby(['基础模型', '场景', '是否缓解', 'rounds'])['total_score'].agg(
        均值='mean',
        方差=lambda x: calculate_variance(x.tolist(), x.mean()),
        样本数='count'
    ).reset_index()

    # 转换为透视表格式：行是模型配置，列是轮次，值包含均值和方差
    model_pivot = model_stat.pivot(
        index=['基础模型', '场景', '是否缓解'],
        columns='rounds',
        values=['均值', '方差']
    )

    # --- 工作表2：6个偏见维度 - 分轮次统计 ---
    dim_data = []
    for _, row in df.iterrows():
        for dim in BIAS_DIMENSIONS:
            dim_data.append({
                '基础模型': row['基础模型'],
                '场景': row['场景'],
                '是否缓解': row['是否缓解'],
                '轮次': row['rounds'],
                '偏见维度': dim,
                '维度得分': row['bias_dim_scores'][dim]
            })

    df_dim = pd.DataFrame(dim_data)
    dim_stat = df_dim.groupby(['偏见维度', '基础模型', '场景', '是否缓解', '轮次'])['维度得分'].agg(
        平均分='mean',
        方差=lambda x: calculate_variance(x.tolist(), x.mean()),
        样本数='count'
    ).reset_index()

    # 保存
    with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
        model_pivot.to_excel(writer, sheet_name='模型-轮次总分统计')
        dim_stat.to_excel(writer, sheet_name='6大偏见维度统计', index=False)

    print(f"✅ 统计完成！结果已计算方差并保存至: {OUTPUT_EXCEL}")


if __name__ == "__main__":
    data_rows = process_all_data()
    if data_rows:
        generate_reports(data_rows)
    else:
        print("❌ 未提取到任何有效数据，请检查 FOLDER_PATH 路径。")