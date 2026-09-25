import json
import os

def read_and_parse_jsonl(file_path):
    """
    读取并解析 JSONL 文件，处理解析异常
    
    Args:
        file_path (str): JSONL 文件路径
    
    Returns:
        dict: 包含成功解析的数据和失败的行信息
    """
    # 初始化结果存储
    result = {
        "success": [],  # 成功解析的行
        "failed": []    # 解析失败的行
    }
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在")
        return result
    
    # 逐行读取并解析
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            # 去除行首尾的空白字符
            line = line.strip()
            if not line:  # 跳过空行
                continue
            
            try:
                # 解析 JSON 数据
                json_data = json.loads(line)
                result["success"].append({
                    "line_number": line_num,
                    "data": json_data
                })
                print(f"第 {line_num} 行解析成功")
                
            except json.JSONDecodeError as e:
                # 捕获 JSON 解析错误
                result["failed"].append({
                    "line_number": line_num,
                    "error": str(e),
                    "raw_data": line
                })
                print(f"第 {line_num} 行解析失败: {e}")
                
            except Exception as e:
                # 捕获其他异常
                result["failed"].append({
                    "line_number": line_num,
                    "error": f"未知错误: {str(e)}",
                    "raw_data": line
                })
                print(f"第 {line_num} 行处理出错: {e}")
    
    return result

def get_completed_10round_ids(parse_result):
    """
    从解析结果中提取所有轮次为10的topic_id，返回去重后的列表
    
    Args:
        parse_result (dict): read_and_parse_jsonl 函数的返回结果
    
    Returns:
        list: 所有完成10轮的topic_id列表（去重）
    """
    completed_ids = set()  # 用集合去重，最后转列表
    
    # 遍历所有成功解析的数据
    for item in parse_result["success"]:
        data = item["data"]
        # 容错：防止字段缺失导致报错
        meta = data.get("meta", {})
        rounds = meta.get("rounds", 0)
        topic_id = meta.get("topic_id")
        
        # 仅收集轮次=10且有有效topic_id的记录
        if rounds == 10 and topic_id:
            completed_ids.add(topic_id)
    
    # 转成列表返回
    completed_ids_list = list(completed_ids)
    print(f"\n提取到 {len(completed_ids_list)} 个完成10轮的topic_id")
    return completed_ids_list

def main():
    # 替换为你的 JSONL 文件路径
    jsonl_file_path = "bias_experiment_results.jsonl"
    
    # 1. 读取并解析文件
    parse_result = read_and_parse_jsonl(jsonl_file_path)
    
    # 2. 输出解析统计信息
    print(f"\n解析完成：")
    print(f"- 成功解析行数：{len(parse_result['success'])}")
    print(f"- 解析失败行数：{len(parse_result['failed'])}")
    
    # 3. 提取所有10轮的topic_id（核心功能）
    completed_10round_ids = get_completed_10round_ids(parse_result)
    print(f"\n完成10轮的topic_id列表：")
    for idx, topic_id in enumerate(completed_10round_ids, 1):
        print(f"{idx}. {topic_id}")
    
    # 4. 修复原示例的语法错误，访问第一个成功解析的数据
    if parse_result['success']:
        # 原错误：parse_result['success'][:]['data'] → 正确写法如下
        first_data = parse_result['success'][0]['data']
        print(f"\n第一个成功解析的数据：")
        print(f"- model: {first_data['meta'].get('model', '无')}")
        print(f"- rounds: {first_data['meta'].get('rounds', '无')}")
        print(f"- topic_id: {first_data['meta'].get('topic_id', '无')}")
        # 容错：防止bias_report字段缺失
        bias_report = first_data.get('bias_report', {})
        print(f"- error: {bias_report.get('error', '无')}")
        raw_content = bias_report.get('raw', '')
        print(f"- raw: {raw_content[:100]}..." if raw_content else "- raw: 无")

if __name__ == "__main__":
    main()