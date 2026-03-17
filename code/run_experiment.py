import json
import os

import bias_experiment
from bias_framework import TargetAgent, ProvocateurAgent, BiasExpertEvaluator, MultiAgentArena
import config
import data_analyze


def load_cmv_data(file_path="cmv_data.jsonl"):
    """
    从本地 JSONL 文件加载 Change-My-View 数据
    返回一个包含 ID, Topic 和 Initial_Context 的列表
    """
    dataset = []

    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return dataset

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                # 提取 prompt 列表中的 content
                # 原始数据格式中 content 包含了 "CMV: Topic\nBackground"
                raw_content = data["prompt"][0]["content"]

                # 简单的解析逻辑：提取标题和正文
                if "CMV:" in raw_content:
                    # 按照换行符分割，第一行通常是标题 (Topic)
                    parts = raw_content.split('\n', 1)
                    topic = parts[0].replace("CMV:", "").strip()
                    context = parts[1].strip() if len(parts) > 1 else topic
                else:
                    topic = raw_content[:50] + "..."  # 兜底方案
                    context = raw_content

                dataset.append({
                    "id": f"cmv_{len(dataset) + 1:03d}",
                    "topic": topic,
                    "initial_context": context,
                    "original_label": data.get("label")  # 保留原始标签供参考
                })
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"解析行失败: {e}")
    print(dataset)
    return dataset


def main():
    # 1. 准备数据集和评估器
    dataset = load_cmv_data(config.Dataset_path)

    #evaluator = BiasExpertEvaluator()

    results_file = "bias_experiment_results_GPT_multiagent.jsonl"
    max_rounds = 10
    i = 0
    device = "cuda"
    parse_result = bias_experiment.read_and_parse_jsonl(results_file)
    completed_10round_ids = bias_experiment.get_completed_10round_ids(parse_result)
    #parse_result = data_analyze.read_and_parse_jsonl(results_file)
    #completed_10round_ids = data_analyze.get_completed_10round_ids(parse_result)
    # 2. 实验循环矩阵
    for model_path in config.MODELS_TO_TEST:
        for method in ["baseline"]:

            for data in dataset:
                if data['id'] in completed_10round_ids:
                    continue
                print("------------------------------")
                print(data["initial_context"])

                print("------------------------------")
                target_agent = TargetAgent(name="gpt", model_path=model_path, stance=data["initial_context"],
                                           method=method, device=device)

                # 3. 设置对手 (针对多智能体场景，可放多个 ProvocateurAgent)
                # 二智能体辩论：一个对手；多智能体讨论：多个对手
                provocateurs = [
                    ProvocateurAgent(name="doubao", model_path=config.MODELS[0], stance=data["initial_context"],
                                     persona_type="aggressive", device=device)]
                arena = MultiAgentArena(target_agent, provocateurs)
                for rounds in range(max_rounds + 1):
                    print(f"Testing: {model_path} | Method: {method} | Rounds: {rounds} | Topic: {data['id']}")

                    # 4. 开启辩论竞技场

                    transcript = arena.run(r=rounds, max_rounds=max_rounds)
                    if rounds == 0 or rounds == 1 or rounds == 3 or rounds == 5 or rounds == 10:
                        target_response = arena.get_target_responses_for_eval()  # 关键：等号后加赋值内容，缩进对齐
                        # 5. 调用 BiasExpert 评估
                        print("BiasExpert 评估：", target_response)
                        #bias_report = evaluator.evaluate(target_response)
                        bias_report = ""
                        # 6. 保存结果
                        final_record = {
                            "meta": {
                                "model": "辩手GPT",
                                "rounds": rounds,
                                "topic_id": data['id']
                            },
                            "bias_report": bias_report,
                            "target_response":target_response,
                            "transcript": transcript
                        }
                        with open(results_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(final_record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()