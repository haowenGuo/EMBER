# Server Runbook

服务器：`fat20`

远端项目根目录：

```text
/home/haowen/Lab/EMBER_SFT_RL_LAB
```

远端 Python：

```text
/home/haowen/Lab/enter/envs/ai/bin/python
```

## 1. Sync Final Experiment Folder

在本地 PowerShell 执行：

```powershell
rsync -av F:/lab/LLM_Eval/EMBER_SFT_RL_LAB/final_experiment_ember_300/ fat20:/home/haowen/Lab/EMBER_SFT_RL_LAB/final_experiment_ember_300/
```

如果没有 `rsync`，使用：

```powershell
scp -r F:/lab/LLM_Eval/EMBER_SFT_RL_LAB/final_experiment_ember_300 fat20:/home/haowen/Lab/EMBER_SFT_RL_LAB/
```

同步评估脚本的元数据兼容补丁：

```powershell
scp F:/lab/LLM_Eval/EMBER_SFT_RL_LAB/formal_eval_cmv101_200_fiveway_exact_fullcontext_qwen_from_qwen/run_fiveway_exact_biasexpert_eval.py fat20:/home/haowen/Lab/EMBER_SFT_RL_LAB/formal_eval_cmv101_200_fiveway_exact_fullcontext_qwen_from_qwen/run_fiveway_exact_biasexpert_eval.py
scp F:/lab/LLM_Eval/EMBER_SFT_RL_LAB/formal_eval_cmv101_200_fiveway_exact_fullcontext_llama_from_qwen/run_fiveway_exact_biasexpert_eval.py fat20:/home/haowen/Lab/EMBER_SFT_RL_LAB/formal_eval_cmv101_200_fiveway_exact_fullcontext_llama_from_qwen/run_fiveway_exact_biasexpert_eval.py
```

同步后设置运行脚本权限：

```powershell
ssh fat20 'chmod +x /home/haowen/Lab/EMBER_SFT_RL_LAB/final_experiment_ember_300/scripts/run_*_final_300_local_attacker.sh'
```

## 2. Server Health Check

```powershell
ssh fat20 'hostname; date; nvidia-smi --query-gpu=index,name,memory.free,memory.total,utilization.gpu --format=csv,noheader,nounits; df -h /home/haowen/Lab /data2; pgrep -af "run_fiveway|train_risk_head|train_sft_adapters|train_rl_policy" || true'
```

## 3. Qwen Final 300 Evaluation

当前推荐口径是前 150 条、0-5 轮、三方法：

```powershell
ssh fat20 'bash /home/haowen/Lab/EMBER_SFT_RL_LAB/final_experiment_ember_300/scripts/run_qwen_final150_round05_three_methods_local_attacker.sh'
```

推荐直接运行本目录提供的脚本：

```powershell
ssh fat20 'bash /home/haowen/Lab/EMBER_SFT_RL_LAB/final_experiment_ember_300/scripts/run_qwen_final_300_local_attacker.sh'
```

展开后的等价命令如下：

```bash
cd /home/haowen/Lab/EMBER_SFT_RL_LAB/formal_eval_cmv101_200_fiveway_exact_fullcontext_qwen_from_qwen
mkdir -p /home/haowen/Lab/EMBER_SFT_RL_LAB/logs
nohup env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2 \
  /home/haowen/Lab/enter/envs/ai/bin/python run_fiveway_exact_biasexpert_eval.py \
  --cmv-path /home/haowen/Lab/EMBER_SFT_RL_LAB/final_experiment_ember_300/data/final_ember_topics_300.jsonl \
  --topic-start 1 \
  --topic-end 300 \
  --variants BASE,EMBER-PROMPT,EMBER-AGENT,SFT,SFT+RL \
  --provocateur-engine local \
  --local-provocateur-model-path /data2/guohaowen_data/huggingface_cache/hub/modelscope/Qwen/Qwen3.5-9B \
  --local-provocateur-device cuda:1 \
  --local-provocateur-max-new-tokens 1024 \
  --output-dir outputs/final_ember300_qwen3_4b_qwen3_5_9b_attacker_$(date +%Y%m%d_%H%M%S) \
  > /home/haowen/Lab/EMBER_SFT_RL_LAB/logs/final_ember300_qwen3_4b_qwen3_5_9b_attacker_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $! > /home/haowen/Lab/EMBER_SFT_RL_LAB/logs/final_ember300_qwen3_4b.pid
```

## 4. Llama Final 300 Evaluation

当前推荐口径是前 150 条、0-5 轮、三方法：

```powershell
ssh fat20 'bash /home/haowen/Lab/EMBER_SFT_RL_LAB/final_experiment_ember_300/scripts/run_llama_final150_round05_three_methods_local_attacker.sh'
```

确认 Llama 使用 Qwen 协议训练出的 checkpoint：

```text
/home/haowen/Lab/EMBER_SFT_RL_LAB/runs/llama3_1_8b_instruct_qwen_protocol/sft/best
/home/haowen/Lab/EMBER_SFT_RL_LAB/runs/llama3_1_8b_instruct_qwen_protocol/rl/final
```

推荐直接运行本目录提供的脚本：

```powershell
ssh fat20 'bash /home/haowen/Lab/EMBER_SFT_RL_LAB/final_experiment_ember_300/scripts/run_llama_final_300_local_attacker.sh'
```

展开后的等价命令如下：

```bash
cd /home/haowen/Lab/EMBER_SFT_RL_LAB/formal_eval_cmv101_200_fiveway_exact_fullcontext_llama_from_qwen
mkdir -p /home/haowen/Lab/EMBER_SFT_RL_LAB/logs
nohup env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2 \
  /home/haowen/Lab/enter/envs/ai/bin/python run_fiveway_exact_biasexpert_eval.py \
  --cmv-path /home/haowen/Lab/EMBER_SFT_RL_LAB/final_experiment_ember_300/data/final_ember_topics_300.jsonl \
  --topic-start 1 \
  --topic-end 300 \
  --variants BASE,EMBER-PROMPT,EMBER-AGENT,SFT,SFT+RL \
  --sft-checkpoint-dir /home/haowen/Lab/EMBER_SFT_RL_LAB/runs/llama3_1_8b_instruct_qwen_protocol/sft/best \
  --rl-checkpoint-dir /home/haowen/Lab/EMBER_SFT_RL_LAB/runs/llama3_1_8b_instruct_qwen_protocol/rl/final \
  --provocateur-engine local \
  --local-provocateur-model-path /data2/guohaowen_data/huggingface_cache/hub/modelscope/Qwen/Qwen3.5-9B \
  --local-provocateur-device cuda:1 \
  --local-provocateur-max-new-tokens 1024 \
  --output-dir outputs/final_ember300_llama3_1_8b_qwen3_5_9b_attacker_$(date +%Y%m%d_%H%M%S) \
  > /home/haowen/Lab/EMBER_SFT_RL_LAB/logs/final_ember300_llama3_1_8b_qwen3_5_9b_attacker_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $! > /home/haowen/Lab/EMBER_SFT_RL_LAB/logs/final_ember300_llama3_1_8b.pid
```

## 5. Progress Check

```powershell
ssh fat20 'cd /home/haowen/Lab/EMBER_SFT_RL_LAB && pgrep -af "run_fiveway" || true; ls -lt logs/final_ember300_*.log | head; tail -80 $(ls -t logs/final_ember300_*.log | head -1)'
```

## 6. Copy Results Back

先复制 analysis、manifest 和 scored rows，不复制模型文件：

```powershell
scp -r fat20:/home/haowen/Lab/EMBER_SFT_RL_LAB/formal_eval_cmv101_200_fiveway_exact_fullcontext_qwen_from_qwen/outputs/final_ember300_qwen3_4b_qwen3_5_9b_attacker_*/ F:/lab/LLM_Eval/EMBER_SFT_RL_LAB/final_experiment_ember_300/results/
scp -r fat20:/home/haowen/Lab/EMBER_SFT_RL_LAB/formal_eval_cmv101_200_fiveway_exact_fullcontext_llama_from_qwen/outputs/final_ember300_llama3_1_8b_qwen3_5_9b_attacker_*/ F:/lab/LLM_Eval/EMBER_SFT_RL_LAB/final_experiment_ember_300/results/
```
