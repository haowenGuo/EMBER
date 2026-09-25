# Solution Overview

## Goal

Extend EMBER from prompt-only mitigation to a parameter-efficient mitigation stack that can still answer in adversarial dialogue while reducing emergent bias.

## Core method

The method is:

```text
Base LLM -> Risk Head -> Adapter Bank -> Gated Residual Fusion -> LM Head
```

For each sequence:

1. the frozen base LLM produces hidden states;
2. the risk head predicts a 6-dim bias-risk vector;
3. the adapter bank applies one debiasing adapter per bias dimension;
4. gate values weight each adapter contribution;
5. fused hidden states are sent to the output head.

## Six dimensions

- political
- gender
- ethnic/cultural
- age
- religion
- disability

## Fusion rule

```text
h' = h + sum_i alpha_i * A_i(h)
```

Where:

- `h` is the base hidden state
- `A_i` is adapter `i`
- `alpha_i` is the predicted risk gate in `[0, 1]`

## Training stages

### Stage 1: Risk-head training

Train only the risk head with all labeled EMBER samples.

Signal:

- 6-dim normalized scores derived from `bias_report`

### Stage 2: Adapter SFT

Freeze base model and risk head. Train adapter bank on low-bias responses only.

Signal:

- completion-only language modeling loss
- optional auxiliary penalty on predicted risk

### Stage 3: RL refinement

Freeze base model and risk head. Continue updating adapters only.

Signal:

- negative predicted risk
- stance preservation
- length prior
- refusal penalty

This is implemented as lightweight group policy optimization without a separate value model.

## Why this fits EMBER

EMBER already shows:

- static prompt mitigation helps early turns;
- mitigation decays under sustained adversarial pressure;
- multi-turn history is the real failure mode.

So the mitigation policy must be learned in-context under adversarial history, not just injected as a front prompt.
