import json
import os
from bias_framework import BiasExpertEvaluator  # 你的评估器类
import config  # 你的配置文件


# ===================== 核心工具函数 =====================
def read_jsonl_file(file_path):
    """读取jsonl文件，返回数据列表"""
    data_list = []
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在！")
        return data_list

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                json_data = json.loads(line)
                data_list.append(json_data)
            except json.JSONDecodeError as e:
                print(f"警告：第{line_num}行解析失败，跳过该行 -> {e}")
    print(f"成功读取 {len(data_list)} 条实验结果数据")
    return data_list


def write_jsonl_file(file_path, data_list):
    """将数据列表写入jsonl文件，覆盖原文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for data in data_list:
            json_str = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
            f.write(json_str + '\n')
    print(f"成功写入 {len(data_list)} 条数据到 {file_path}")


def fill_bias_report():
    """主逻辑：读取数据 -> 评估偏见 -> 回填字段"""
    # 1. 初始化文件路径和评估器
    results_file = "bias_experiment_results_GPT.jsonl"
    evaluator = BiasExpertEvaluator()  # 初始化你的偏见评估器
    total_count = 0
    filled_count = 0

    # 2. 读取原有实验结果数据
    data_list = read_jsonl_file(results_file)
    if not data_list:
        return

    # 3. 循环处理每条数据，批量评估+回填
    for idx, single_data in enumerate(data_list):
        total_count += 1
        # 提取当前条目的关键字段
        target_response = single_data.get("target_response", "")
        current_bias_report = single_data.get("bias_report", "")

        # 跳过：无目标回复 / 已有评估报告 的条目（断点续跑核心逻辑）
        if not target_response or current_bias_report.strip():
            continue

        try:
            # ============ 核心调用：你的偏见评估方法 ============
            print(f"\n正在评估第 {idx + 1}/{len(data_list)} 条数据 | topic_id: {single_data['meta']['topic_id']}")
            bias_report = evaluator.evaluate(target_response)
            # ==================================================

            # 回填评估结果到bias_report字段
            single_data["bias_report"] = bias_report
            filled_count += 1
            print(f"✅ 评估完成，已回填bias_report字段")

        except Exception as e:
            print(f"❌ 评估失败：{str(e)} | topic_id: {single_data['meta']['topic_id']}")
            single_data["bias_report"] = f"评估异常: {str(e)}"
            continue

    # 4. 将更新后的数据写回原文件（完整保留所有原有内容）
    write_jsonl_file(results_file, data_list)

    # 5. 打印统计结果
    print("\n=====================================")
    print(f"处理完成 | 总数据条数: {total_count}")
    print(f"本次新增评估条数: {filled_count}")
    print(f"已完成评估条数: {len([d for d in data_list if d.get('bias_report', '').strip()])}")
    print("=====================================")


# ===================== 兼容你的原有工具函数（可选保留） =====================
def read_and_parse_jsonl(file_path):
    """兼容你原代码里的解析方法，防止其他调用报错"""
    data = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    return data


def get_completed_10round_ids(parse_result):
    """兼容你原代码里的方法"""
    completed_ids = set()
    for item in parse_result:
        if item.get("meta", {}).get("rounds") == 10:
            completed_ids.add(item.get("meta", {}).get("topic_id"))
    return completed_ids


# ===================== 程序入口 =====================
if __name__ == "__main__":
    # 直接执行：评估+回填主逻辑
    fill_bias_report()