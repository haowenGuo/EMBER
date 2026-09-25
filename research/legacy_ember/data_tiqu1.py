import json

def extract_round_10_data(input_file, output_file):
    results = []
    count = 0
    
    # 逐行读取 JSONL 数据
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                item = json.loads(line)
                # 核心逻辑：检查 meta 字段中的 rounds 是否为 10
                if item.get("meta", {}).get("rounds") == 10:
                    results.append(item)
                    count += 1
            except json.JSONDecodeError:
                print(f"警告：跳过无效的 JSON 行")

    # 将提取出的第10轮数据保存为新的文件
    with open(output_file, 'w', encoding='utf-8') as f_out:
        # 你可以选择保存为标准的 JSON 列表，也可以继续保存为 JSONL
        # 这里演示保存为标准 JSON 格式，方便查看
        json.dump(results, f_out, ensure_ascii=False, indent=4)

    print(f"--- 提取完成 ---")
    print(f"总计找到第10轮对话记录: {count} 条")
    print(f"结果已保存至: {output_file}")

# --- 请在此处修改你的文件名 ---
input_filename = 'bias_experiment_qwen_mitigation.jsonl'  # 替换成你的原始文件名
output_filename = 'allrounds/bias_experiment_qwen_mitigation.jsonl'

extract_round_10_data(input_filename, output_filename)