import json
from collections import defaultdict
import argparse
from typing import List, Dict, Any


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """加载JSONL文件，跳过解析错误行"""
    data = []
    error_lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                json_obj = json.loads(line)
                data.append({
                    'line_num': line_num,
                    'data': json_obj
                })
            except json.JSONDecodeError as e:
                error_lines.append((line_num, str(e)))
                print(f"⚠️  第{line_num}行解析失败，跳过。错误：{e}")

    # 打印加载统计
    print(f"\n📊 原始文件加载统计：")
    print(f"   总行数（非空）：{len(data) + len(error_lines)}")
    print(f"   有效数据行数：{len(data)}")
    print(f"   解析错误行数：{len(error_lines)}")
    return data


def deduplicate_by_topic_round(
        raw_data: List[Dict[str, Any]],
        topic_key: str = 'meta.topic_id',
        round_key: str = 'meta.rounds',
        keep_strategy: str = 'first'  # first/last
) -> List[Dict[str, Any]]:
    """
    按TOPIC+ROUNDS去重，保证每个TOPIC的每个ROUNDS只留一条数据

    Args:
        raw_data: 加载的原始数据（含行号和数据体）
        topic_key: TOPIC字段路径（如'meta.topic_id'或'topic_id'）
        round_key: ROUNDS字段路径（如'meta.rounds'或'rounds'）
        keep_strategy: 保留策略 - first(第一条)/last(最后一条)

    Returns:
        去重后的数据列表（仅包含data部分）
    """
    # 构建去重索引：key = f"{topic_id}_{rounds}"，value = 数据
    dedup_index = {}

    for item in raw_data:
        json_obj = item['data']
        line_num = item['line_num']

        # 解析TOPIC ID（支持嵌套字段，如meta.topic_id）
        topic_id = json_obj
        for key_part in topic_key.split('.'):
            topic_id = topic_id.get(key_part, '未知TOPIC') if isinstance(topic_id, dict) else '未知TOPIC'

        # 解析ROUNDS（支持嵌套字段，如meta.rounds）
        rounds = json_obj
        for key_part in round_key.split('.'):
            rounds = rounds.get(key_part, -1) if isinstance(rounds, dict) else -1
        # 确保rounds是整数
        try:
            rounds = int(rounds)
        except (ValueError, TypeError):
            rounds = -1

        # 生成去重唯一键
        dedup_key = f"{topic_id}_{rounds}"

        # 根据策略保留数据
        if dedup_key not in dedup_index:
            # 首次出现，直接保存
            dedup_index[dedup_key] = item
        else:
            # 重复数据，按策略替换
            if keep_strategy == 'last':
                dedup_index[dedup_key] = item

    # 提取去重后的数据（仅保留data部分）
    deduped_data = [v['data'] for v in dedup_index.values()]

    # 打印去重统计
    print(f"\n🔍 去重统计（按TOPIC+ROUNDS）：")
    print(f"   原始有效数据条数：{len(raw_data)}")
    print(f"   去重后数据条数：{len(deduped_data)}")
    print(f"   重复数据条数：{len(raw_data) - len(deduped_data)}")
    print(f"   保留策略：{keep_strategy}（{'第一条' if keep_strategy == 'first' else '最后一条'}）")

    return deduped_data


def save_jsonl(data: List[Dict[str, Any]], output_path: str):
    """保存去重后的数据到JSONL文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for json_obj in data:
            f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
    print(f"\n✅ 去重后文件已保存：{output_path}")


def main():
    # 命令行参数配置（也可直接修改下方默认值）
    parser = argparse.ArgumentParser(description='JSONL文件去重：按TOPIC+ROUNDS保留唯一数据')
    parser.add_argument('--input', '-i', default='bias_experiment_results_GPT_mitigation_evaluated.jsonl',
                        help='输入JSONL文件路径（默认：bias_experiment_results_mitigation.jsonl）')
    parser.add_argument('--output', '-o', default='bias_experiment_results_GPT_mitigation_evaluated.jsonl',
                        help='输出去重后JSONL文件路径（默认：bias_experiment_results_deduplicated.jsonl）')
    parser.add_argument('--topic-key', default='meta.topic_id',
                        help='TOPIC字段路径（如meta.topic_id/topic_id，默认：meta.topic_id）')
    parser.add_argument('--round-key', default='meta.rounds',
                        help='ROUNDS字段路径（如meta.rounds/rounds，默认：meta.rounds）')
    parser.add_argument('--keep', '-k', default='last', choices=['first', 'last'],
                        help='重复数据保留策略（first=第一条/last=最后一条，默认：first）')

    args = parser.parse_args()

    # 1. 加载原始数据
    print("🔄 开始加载原始JSONL文件...")
    raw_data = load_jsonl(args.input)
    if not raw_data:
        print("❌ 无有效数据，脚本退出")
        return

    # 2. 按TOPIC+ROUNDS去重
    print("\n🔄 开始按TOPIC+ROUNDS去重...")
    deduped_data = deduplicate_by_topic_round(
        raw_data=raw_data,
        topic_key=args.topic_key,
        round_key=args.round_key,
        keep_strategy=args.keep
    )

    # 3. 保存去重后的数据
    print("\n🔄 开始保存去重后文件...")
    save_jsonl(deduped_data, args.output)

    print("\n🎉 去重脚本执行完成！")


if __name__ == "__main__":
    # 运行方式1：命令行（推荐）
    # python dedup_jsonl.py -i 你的输入文件.jsonl -o 输出文件.jsonl --keep last
    #
    # 运行方式2：直接修改默认参数后运行
    main()