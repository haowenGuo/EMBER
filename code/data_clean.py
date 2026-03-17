import json
import argparse
from typing import List, Dict, Any


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """加载JSONL文件，跳过解析错误行/空行"""
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

    # 加载统计日志
    print(f"\n📊 原始文件加载统计：")
    print(f"   总行数（非空）：{len(data) + len(error_lines)}")
    print(f"   有效数据行数：{len(data)}")
    print(f"   解析错误行数：{len(error_lines)}")
    return data


def filter_topic_id_cmv_001_to_100(raw_data: List[Dict[str, Any]], topic_key: str = 'meta.topic_id') -> List[
    Dict[str, Any]]:
    """
    ✅ 核心重构：精准过滤，只保留 topic_id 为 cmv_001 ~ cmv_100 的数据
    适配格式：cmv_001、cmv_002 ... cmv_099、cmv_100 （带前置零的三位数格式）
    过滤规则：
        1. 必须是 cmv_ 开头的字符串
        2. 下划线后是 001~100 的数字（含前置零、纯数字）
        3. 不符合格式/超出范围的全部过滤
    """
    filtered_data = []
    filtered_out_count = 0

    for item in raw_data:
        json_obj = item['data']
        # 解析嵌套/根节点的topic_id
        topic_id = json_obj
        for key_part in topic_key.split('.'):
            if isinstance(topic_id, dict):
                topic_id = topic_id.get(key_part, None)
            else:
                topic_id = None
                break

        # 开始严格校验topic_id格式
        if isinstance(topic_id, str) and topic_id.startswith("cmv_"):
            try:
                # 截取 cmv_ 后的数字部分：cmv_001 → 001 ; cmv_100 →100
                num_part = topic_id.split("_")[-1]
                # 转为整数，自动消除前置零：001→1，099→99，100→100
                topic_num = int(num_part)
                # ✅ 核心过滤条件：只保留 1 ≤ 数字 ≤ 100
                if 1 <= topic_num <= 100:
                    filtered_data.append(item)
                else:
                    filtered_out_count += 1
            except (ValueError, IndexError):
                # 数字部分转整数失败/格式错误，过滤
                filtered_out_count += 1
        else:
            # 不是cmv_开头/空值/非字符串，过滤
            filtered_out_count += 1

    # 过滤统计日志
    print(f"\n⚡ TOPIC过滤统计（仅保留 cmv_001 ~ cmv_100）：")
    print(f"   过滤前数据条数：{len(raw_data)}")
    print(f"   过滤后数据条数：{len(filtered_data)}")
    print(f"   过滤剔除条数  ：{filtered_out_count}")
    return filtered_data


def deduplicate_by_topic_round(
        raw_data: List[Dict[str, Any]],
        topic_key: str = 'meta.topic_id',
        round_key: str = 'meta.rounds',
        keep_strategy: str = 'last'
) -> List[Dict[str, Any]]:
    """按【topic_id + rounds】组合键去重，同话题同轮次仅保留1条"""
    dedup_index = {}
    for item in raw_data:
        json_obj = item['data']
        # 解析topic_id
        topic_id = json_obj
        for key_part in topic_key.split('.'):
            topic_id = topic_id.get(key_part, '未知TOPIC') if isinstance(topic_id, dict) else '未知TOPIC'
        # 解析rounds并强转整数
        rounds = json_obj
        for key_part in round_key.split('.'):
            rounds = rounds.get(key_part, -1) if isinstance(rounds, dict) else -1
        try:
            rounds = int(rounds)
        except (ValueError, TypeError):
            rounds = -1

        dedup_key = f"{topic_id}_{rounds}"
        # 按策略保留：首次出现直接存，重复则按last/first覆盖
        if dedup_key not in dedup_index or keep_strategy == 'last':
            dedup_index[dedup_key] = item

    deduped_data = [v['data'] for v in dedup_index.values()]
    # 去重统计日志
    print(f"\n🔍 去重统计（按 TOPIC_ID + ROUNDS 组合键）：")
    print(f"   去重前数据条数：{len(raw_data)}")
    print(f"   去重后数据条数：{len(deduped_data)}")
    print(f"   剔除重复条数  ：{len(raw_data) - len(deduped_data)}")
    print(f"   重复保留策略  ：{keep_strategy} → {'保留最新最后1条' if keep_strategy == 'last' else '保留最早第1条'}")
    return deduped_data


def save_jsonl(data: List[Dict[str, Any]], output_path: str):
    """保存JSONL文件，中文/特殊字符正常显示，无乱码"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for json_obj in data:
            f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
    print(f"\n✅ 最终文件已保存：{output_path}")
    print(f"✅ 最终有效数据总行数：{len(data)}")


def main():
    parser = argparse.ArgumentParser(description='JSONL数据处理：过滤[cmv_001~cmv_100] + TOPIC+ROUND去重')
    parser.add_argument('--input', '-i',
                        default='bias_experiment_qwen_mitigation.jsonl',
                        help='输入JSONL文件路径')
    parser.add_argument('--output', '-o',
                        default='experiment_data/bias_experiment_qwen_mitigation.jsonl',
                        help='输出最终结果文件路径，默认自动加后缀防覆盖')
    parser.add_argument('--topic-key', default='meta.topic_id',
                        help='topic_id字段路径，如根节点写topic_id，嵌套写meta.topic_id')
    parser.add_argument('--round-key', default='meta.rounds',
                        help='rounds字段路径，如根节点写rounds，嵌套写meta.rounds')
    parser.add_argument('--keep', '-k', default='last', choices=['first', 'last'],
                        help='重复数据保留策略：first=第一条，last=最后一条（推荐）')

    args = parser.parse_args()

    # 执行流程：加载 → 过滤 → 去重 → 保存 【最优顺序，性能最大化】
    raw_data = load_jsonl(args.input)
    if not raw_data:
        print("❌ 原始文件无有效数据，程序退出！")
        return

    filtered_data = filter_topic_id_cmv_001_to_100(raw_data, args.topic_key)
    if not filtered_data:
        print("❌ 过滤后无符合条件的数据，程序退出！")
        return

    deduped_data = deduplicate_by_topic_round(filtered_data, args.topic_key, args.round_key, args.keep)
    save_jsonl(deduped_data, args.output)

    print("\n🎉 【过滤+去重】全流程执行完成！")


if __name__ == "__main__":
    # 运行方式1：直接运行（使用默认参数）
    # 运行方式2：命令行自定义参数
    # python 脚本名.py -i 你的输入文件.jsonl -o 你的输出文件.jsonl --keep last
    main()