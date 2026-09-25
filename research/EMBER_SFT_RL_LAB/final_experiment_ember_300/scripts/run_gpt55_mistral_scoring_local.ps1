param(
    [string]$RepoRoot = "F:\lab\LLM_Eval\EMBER_SFT_RL_LAB",
    [string]$LegacyConfig = "F:\lab\LLM_Eval\EMBER_ARR_REVISION_LAB\original_ember\top_level_py\config.py",
    [string]$BaseUrl = "https://api.gptsapi.net/v1",
    [string]$Model = "gpt-5.5",
    [string]$ModelLabel = "gpt55",
    [int]$Concurrency = 6
)

$ErrorActionPreference = "Stop"

$ScriptPath = Join-Path $RepoRoot "final_experiment_ember_300\scripts\openai_score_biasexpert_prompt.py"
$BiasPromptSource = Join-Path $RepoRoot "formal_eval_cmv101_200_fiveway_exact_fullcontext_qwen_from_qwen\run_fiveway_exact_biasexpert_eval.py"
$ResultsRoot = Join-Path $RepoRoot "final_experiment_ember_300\results"
$OutputPrefix = "codex_${ModelLabel}_simplified_v2"

$Tasks = @(
    @{
        Name = "mistral_first_eval_dual"
        Input = Join-Path $ResultsRoot "vllm_first_eval150_round05_mistral_20260625_164408\mistral\scored_rounds.jsonl"
        Output = Join-Path $ResultsRoot "${OutputPrefix}_first_eval_mistral_full_20260627\mistral"
    },
    @{
        Name = "mistral_first_eval_multiagent"
        Input = Join-Path $ResultsRoot "vllm_first_eval150_round05_multiagent_mistral_20260626_083946\mistral\scored_rounds.jsonl"
        Output = Join-Path $ResultsRoot "${OutputPrefix}_first_eval_multiagent_mistral_full_20260627\mistral"
    }
)

foreach ($Task in $Tasks) {
    if (-not (Test-Path -LiteralPath $Task.Input)) {
        throw "Missing input jsonl: $($Task.Input)"
    }

    New-Item -ItemType Directory -Force -Path $Task.Output | Out-Null
    Write-Host ("{0} START {1}" -f (Get-Date -Format s), $Task.Name)

    python $ScriptPath `
        --input-jsonl $Task.Input `
        --output-dir $Task.Output `
        --bias-prompt-source $BiasPromptSource `
        --legacy-config $LegacyConfig `
        --model $Model `
        --model-label $ModelLabel `
        --api-mode chat `
        --base-url $BaseUrl `
        --concurrency $Concurrency `
        --max-output-tokens 5000 `
        --temperature 0.3 `
        --top-p 0.95 `
        --request-timeout 240 `
        --max-retries 3

    if ($LASTEXITCODE -ne 0) {
        throw "Scoring failed for $($Task.Name) with exit code $LASTEXITCODE"
    }

    Write-Host ("{0} DONE {1}" -f (Get-Date -Format s), $Task.Name)
}

Write-Host ("{0} ALL_DONE" -f (Get-Date -Format s))
