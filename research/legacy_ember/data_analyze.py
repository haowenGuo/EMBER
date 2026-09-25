import json
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Any

# 定义需要保留的关键轮次（核心配置）
TARGET_ROUNDS = [0, 1, 3, 5, 10]


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """加载JSONL文件，跳过解析错误行"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                json_obj = json.loads(line)
                data.append(json_obj)
            except json.JSONDecodeError as e:
                print(f"警告：第{line_num}行解析失败，跳过。错误：{e}")
    return data


def extract_only_bias_level(json_obj: Dict[str, Any]) -> Dict[str, Any]:
    """仅提取核心数据：topic_id + rounds + 6个维度偏见等级"""
    meta = json_obj.get('meta', {})
    topic_id = meta.get('topic_id', '未知')
    rounds = int(meta.get('rounds', -1))

    # 仅解析偏见等级
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

    # 仅保留6个维度的等级
    bias_levels = {
        '政治': bias_analysis.get('political', {}).get('level', 'None'),
        '性别': bias_analysis.get('gender', {}).get('level', 'None'),
        '种族/文化': bias_analysis.get('ethnic_cultural', {}).get('level', 'None'),
        '年龄': bias_analysis.get('age', {}).get('level', 'None'),
        '宗教': bias_analysis.get('religion', {}).get('level', 'None'),
        '残障': bias_analysis.get('disability', {}).get('level', 'None')
    }

    return {
        'topic_id': topic_id,
        'rounds': rounds,
        'bias_levels': bias_levels
    }


def filter_target_rounds(topic_data: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, str]]:
    """仅保留目标轮次（0/1/3/5/10），缺失的标注「无数据」"""
    filtered = {}
    for rounds in TARGET_ROUNDS:
        if rounds in topic_data and topic_data[rounds]['topic_id'] != '未知':
            filtered[rounds] = topic_data[rounds]['bias_levels']
        else:
            filtered[rounds] = {
                '政治': '无数据', '性别': '无数据', '种族/文化': '无数据',
                '年龄': '无数据', '宗教': '无数据', '残障': '无数据'
            }
    return filtered


def analyze_trend_target(level_dict: Dict[int, Dict[str, str]]) -> Dict[str, str]:
    """仅分析目标轮次间的趋势变化"""
    level_score = {'无数据': -1, 'None': 0, 'Low': 1, 'Moderate': 2, 'High': 3}
    trend = {}

    # 按目标轮次顺序对比（0→1, 1→3, 3→5, 5→10）
    for i in range(len(TARGET_ROUNDS) - 1):
        curr_round = TARGET_ROUNDS[i]
        next_round = TARGET_ROUNDS[i + 1]
        round_pair = f"R{curr_round}→R{next_round}"

        curr_levels = level_dict[curr_round]
        next_levels = level_dict[next_round]

        changes = []
        for dim in ['政治', '性别', '种族/文化', '年龄', '宗教', '残障']:
            curr = curr_levels[dim]
            next_ = next_levels[dim]

            if curr == '无数据' and next_ == '无数据':
                continue
            if curr == '无数据':
                changes.append(f"{dim}: 新增({next_})")
            elif next_ == '无数据':
                changes.append(f"{dim}: 缺失({curr})")
            else:
                curr_s = level_score[curr]
                next_s = level_score[next_]
                if next_s > curr_s:
                    changes.append(f"{dim}: ↑({curr}→{next_})")
                elif next_s < curr_s:
                    changes.append(f"{dim}: ↓({curr}→{next_})")

        trend[round_pair] = '; '.join(changes) if changes else '无变化'
    return trend


def print_core_report(all_topic_data: Dict[str, Dict[int, Dict[str, str]]]):
    """打印核心报表：仅目标轮次的等级+趋势"""
    print("=" * 120)
    print(f"【核心偏见等级报表（仅保留 rounds {TARGET_ROUNDS}）】")
    print("=" * 120)

    # 1. 打印每个TOPIC的等级表（仅目标轮次）
    for topic_id, level_dict in all_topic_data.items():
        print(f"\n📌 TOPIC ID: {topic_id}")
        # 表头
        print("-" * 120)
        header = f"{'ROUNDS':<6} | {'政治':<8} | {'性别':<8} | {'种族/文化':<10} | {'年龄':<8} | {'宗教':<8} | {'残障':<8}"
        print(header)
        print("-" * 120)
        # 仅打印目标轮次
        for rounds in TARGET_ROUNDS:
            levels = level_dict[rounds]
            row = (
                f"{rounds:<6} | "
                f"{levels['政治']:<8} | "
                f"{levels['性别']:<8} | "
                f"{levels['种族/文化']:<10} | "
                f"{levels['年龄']:<8} | "
                f"{levels['宗教']:<8} | "
                f"{levels['残障']:<8}"
            )
            print(row)
        print("-" * 120)

        # 2. 打印该TOPIC的趋势变化（仅目标轮次间）
        trend = analyze_trend_target(level_dict)
        print("🔄 等级变化趋势（仅目标轮次）：")
        for round_pair, change in trend.items():
            if change != '无变化':
                print(f"  {round_pair}: {change}")
        print("=" * 120)


def generate_excel_target(all_topic_data: Dict[str, Dict[int, Dict[str, str]]], output_path: str):
    """生成仅目标轮次的极简Excel"""
    rows = []
    for topic_id, level_dict in all_topic_data.items():
        for rounds in TARGET_ROUNDS:
            levels = level_dict[rounds]
            rows.append({
                'TOPIC_ID': topic_id,
                'ROUNDS': rounds,
                '政治偏见等级': levels['政治'],
                '性别偏见等级': levels['性别'],
                '种族/文化偏见等级': levels['种族/文化'],
                '年龄偏见等级': levels['年龄'],
                '宗教偏见等级': levels['宗教'],
                '残障偏见等级': levels['残障']
            })

    # 写入Excel并排序
    df = pd.DataFrame(rows)
    df = df.sort_values(by=['TOPIC_ID', 'ROUNDS'])
    df.to_excel(output_path, index=False, engine='openpyxl')
    print(f"\n✅ 极简Excel已生成：{output_path}")


def main():
    # --------------------------
    # 请修改这两行的文件路径！
    # --------------------------
    jsonl_path = "bias_experiment_results_llama_mitigation.jsonl"  # 你的JSONL文件路径
    excel_path = "bias_experiment_results_llama_mitigation_analyze.xlsx"  # 输出Excel路径

    # 1. 加载数据
    print("🔍 加载JSONL文件...")
    json_data = load_jsonl(jsonl_path)
    if not json_data:
        print("❌ 无有效数据！")
        return

    # 2. 按TOPIC分组，仅保留等级数据
    topic_dict = defaultdict(dict)
    for obj in json_data:
        bias_info = extract_only_bias_level(obj)
        topic_id = bias_info['topic_id']
        rounds = bias_info['rounds']
        if rounds in TARGET_ROUNDS and topic_id != '未知':
            topic_dict[topic_id][rounds] = bias_info

    # 3. 筛选目标轮次，生成核心数据
    core_data = {}
    for topic_id, data in topic_dict.items():
        core_data[topic_id] = filter_target_rounds(data)

    # 4. 打印核心报表
    print_core_report(core_data)

    # 5. 生成Excel
    generate_excel_target(core_data, excel_path)
    print("\n🎉 报表生成完成！仅保留 rounds 0/1/3/5/10 数据")


if __name__ == "__main__":
    # 首次运行安装依赖
    # pip install pandas openpyxl
    main()