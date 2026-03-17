import json
import os
import concurrent.futures
import threading
from ember_multiagent import TargetAgent, ProvocateurAgent, BiasExpertEvaluator, MultiAgentArena
import config
import bias_experiment

# ================= 全局核心配置 =================
THREAD_NUM = 20  # 建议根据显存调整，20个线程跑4个模型实例压力非常大
DATA_LIMIT = 100
results_file = "bias_experiment_results_GPT_multiagent2.jsonl"
max_rounds = 10
device = "cuda:0"

# 文件写入锁，防止多线程同时写入导致JSON损坏
write_lock = threading.Lock()


def get_completed_topics(file_path):
    """
    ✅ 核心逻辑：扫描结果文件，只有拥有 round 10 记录的 topic_id 才被视为已完成
    """
    completed_ids = set()
    if not os.path.exists(file_path):
        return completed_ids

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                # 检查 meta 里的 rounds 是否等于我们设定的最大轮数
                if data.get("meta", {}).get("rounds") == max_rounds:
                    tid = data.get("meta", {}).get("topic_id")
                    if tid:
                        completed_ids.add(tid)
            except:
                continue
    return completed_ids


def run_experiment_batch(batch_data):
    """
    单线程执行逻辑
    """
    # 每次启动前获取已完成名单（round 10 标准）
    completed_ids = get_completed_topics(results_file)

    for data in batch_data:
        # ✅ 判断：如果没有 round 10 的数据，就重跑整个话题
        if data['id'] in completed_ids:
            print(f"⏩ Topic {data['id']} 已有 Round {max_rounds} 记录，跳过。")
            continue

        print(f"\n🔥 正在启动 Topic: {data['id']} (重新开始或续跑)")

        try:
            # 初始化智能体
            target_agent = TargetAgent(
                name="GPT",
                model_path=config.MODELS_TO_TEST[0],
                stance=data["initial_context"],
                method="baseline",
                device=device
            )

            provocateurs = [ProvocateurAgent(name="d0", helpermodel_name1="d1", helpermodel_name2="d2",
                                             targetmodel_name=target_agent.model_name, model_path=config.MODELS[0],
                                             stance=data["initial_context"], persona_type="aggressive", device=device),
                            ProvocateurAgent(name="d1", helpermodel_name1="d0", helpermodel_name2="d2",
                                             targetmodel_name=target_agent.model_name, model_path=config.MODELS[0],
                                             stance=data["initial_context"], persona_type="aggressive", device=device),
                            ProvocateurAgent(name="d2", helpermodel_name1="d0", helpermodel_name2="d1",
                                             targetmodel_name=target_agent.model_name, model_path=config.MODELS[0],
                                             stance=data["initial_context"], persona_type="aggressive",
                                             device=device), ]

            arena = MultiAgentArena(target_agent, provocateurs)

            # 运行辩论
            for r in range(max_rounds + 1):
                # 只有特定的轮次需要保存
                if r in [0, 1, 3, 5, 10]:
                    # 检查文件里是否已经有了这一轮的数据（防止重跑时 round 0/1 重复写入）
                    # 这一步是可选的，如果允许文件里有重复的中间过程则可以简化

                    transcript = arena.run(r=r, max_rounds=max_rounds)
                    target_response = arena.get_target_responses_for_eval()

                    final_record = {
                        "meta": {
                            "model": "辩手GPT",
                            "rounds": r,
                            "topic_id": data['id']
                        },
                        "bias_report": "",
                        "target_response": target_response,
                        "transcript": transcript
                    }

                    # ✅ 加锁写入，确保 JSONL 格式绝对正确
                    with write_lock:
                        with open(results_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(final_record, ensure_ascii=False) + "\n")

                    print(f"✅ Topic {data['id']} | Round {r} 已保存")

        except Exception as e:
            print(f"❌ Topic {data['id']} 运行出错: {e}")
            continue

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
    # 1. 加载数据
    dataset = load_cmv_data(config.Dataset_path)[:DATA_LIMIT]

    # 2. 获取已经跑完 round 10 的 ID
    done = get_completed_topics(results_file)
    todo_dataset = [d for d in dataset if d['id'] not in done]

    print(f"📊 统计: 总需跑 {len(dataset)} 条 | 已完成(Round 10) {len(done)} 条 | 待处理 {len(todo_dataset)} 条")

    if not todo_dataset:
        print("🎉 所有任务已达到 Round 10，无需运行。")
        return

    # 3. 重新切分任务给线程
    chunk_size = (len(todo_dataset) + THREAD_NUM - 1) // THREAD_NUM
    batches = [todo_dataset[i:i + chunk_size] for i in range(0, len(todo_dataset), chunk_size)]

    # 4. 并行启动
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_NUM) as executor:
        executor.map(run_experiment_batch, batches)


if __name__ == "__main__":
    main()