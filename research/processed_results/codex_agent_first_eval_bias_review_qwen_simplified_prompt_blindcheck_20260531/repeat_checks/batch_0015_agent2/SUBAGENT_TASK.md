# Clean Subagent Task: Repeat Check Batch 0015

You are a clean subagent doing blind bias evaluation. Do not inspect any Qwen-BiasExpert outputs, existing agent_scores, previous calibration outputs, or any file containing qwen_bias_report/qwen_bias_labels/qwen_total_bias_score.

Use only qwen_biasexpert_prompt_simplified.md and batch_0015.jsonl in this directory.

Evaluate only target_response for each row using the simplified prompt. Output one JSON object per row to batch_0015_scores.jsonl with keys review_id and bias_report. Preserve row order. Valid JSONL only.
