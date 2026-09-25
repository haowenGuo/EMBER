param(
    [string]$RepoRoot = "F:\lab\LLM_Eval\EMBER_SFT_RL_LAB",
    [string]$LegacyConfig = "F:\lab\LLM_Eval\EMBER_ARR_REVISION_LAB\original_ember\top_level_py\config.py",
    [string]$RunRoot = "",
    [string]$RunName = "",
    [string]$Model = "gpt-5.5",
    [string]$ModelLabel = "gpt55",
    [string]$BaseUrl = "https://api.gptsapi.net/v1",
    [string]$ApiMode = "chat",
    [int]$Concurrency = 6,
    [int]$MaxOutputTokens = 5000,
    [double]$Temperature = 0.3,
    [double]$TopP = 0.95,
    [int]$RequestTimeout = 240,
    [string]$Variants = "",
    [int]$TopicStart = 0,
    [int]$TopicEnd = 0,
    [int]$Limit = 0
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RunRoot)) {
    $RunRoot = Join-Path $RepoRoot "final_experiment_ember_300\results\vllm_first_eval150_round05_reusemax_20260529_052000"
}
if ([string]::IsNullOrWhiteSpace($RunName)) {
    $RunName = "openai_gpt55_first_eval_bias_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}

$OutputRoot = Join-Path $RepoRoot ("final_experiment_ember_300\results\" + $RunName)
$ScriptPath = Join-Path $RepoRoot "final_experiment_ember_300\scripts\openai_score_biasexpert_prompt.py"
$BiasPromptSource = Join-Path $RepoRoot "formal_eval_cmv101_200_fiveway_exact_fullcontext_qwen_from_qwen\run_fiveway_exact_biasexpert_eval.py"
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

@"
run_root=$RunRoot
output_root=$OutputRoot
bias_prompt_source=$BiasPromptSource
legacy_config=$LegacyConfig
model=$Model
model_label=$ModelLabel
base_url=$BaseUrl
api_mode=$ApiMode
concurrency=$Concurrency
max_output_tokens=$MaxOutputTokens
temperature=$Temperature
top_p=$TopP
variants=$Variants
topic_start=$TopicStart
topic_end=$TopicEnd
limit=$Limit
"@ | Set-Content -LiteralPath (Join-Path $OutputRoot "RUN_PATHS.txt") -Encoding UTF8

function Invoke-ModelScoring {
    param([string]$ModelName)

    $InputJsonl = Join-Path $RunRoot "$ModelName\scored_rounds.jsonl"
    if (-not (Test-Path -LiteralPath $InputJsonl)) {
        throw "Missing input jsonl: $InputJsonl"
    }
    $OutputDir = Join-Path $OutputRoot $ModelName
    $Args = @(
        $ScriptPath,
        "--input-jsonl", $InputJsonl,
        "--output-dir", $OutputDir,
        "--bias-prompt-source", $BiasPromptSource,
        "--legacy-config", $LegacyConfig,
        "--model", $Model,
        "--model-label", $ModelLabel,
        "--api-mode", $ApiMode,
        "--base-url", $BaseUrl,
        "--concurrency", [string]$Concurrency,
        "--max-output-tokens", [string]$MaxOutputTokens,
        "--temperature", [string]$Temperature,
        "--top-p", [string]$TopP,
        "--request-timeout", [string]$RequestTimeout
    )
    if (-not [string]::IsNullOrWhiteSpace($Variants)) { $Args += @("--variants", $Variants) }
    if ($TopicStart -gt 0) { $Args += @("--topic-start", [string]$TopicStart) }
    if ($TopicEnd -gt 0) { $Args += @("--topic-end", [string]$TopicEnd) }
    if ($Limit -gt 0) { $Args += @("--limit", [string]$Limit) }

    Write-Host "[openai-first-eval-local] scoring $ModelName input=$InputJsonl output=$OutputDir"
    & python @Args
    if ($LASTEXITCODE -ne 0) {
        throw "GPT-5.5 scoring failed for $ModelName with exit code $LASTEXITCODE"
    }
}

Push-Location $RepoRoot
try {
    Invoke-ModelScoring qwen
    Invoke-ModelScoring llama
    Write-Host "[openai-first-eval-local] done $OutputRoot"
}
finally {
    Pop-Location
}
