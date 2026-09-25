import json
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from .utils import unwrap_model


DIMENSION_COUNT = 6


class LowRankAdapter(nn.Module):
    def __init__(self, hidden_size, rank=8, alpha=16.0, dropout=0.05):
        super().__init__()
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scaling = alpha / max(rank, 1)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        return self.up(self.dropout(self.down(hidden_states))) * self.scaling


class AdapterBank(nn.Module):
    def __init__(self, hidden_size, rank=8, alpha=16.0, dropout=0.05):
        super().__init__()
        self.adapters = nn.ModuleList(
            [LowRankAdapter(hidden_size, rank=rank, alpha=alpha, dropout=dropout) for _ in range(DIMENSION_COUNT)]
        )

    def forward(self, hidden_states, gate_values):
        adapter_dtype = self.adapters[0].down.weight.dtype
        source_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(adapter_dtype)
        gate_values = gate_values.to(adapter_dtype)
        fused = hidden_states
        for index, adapter in enumerate(self.adapters):
            gate = gate_values[:, index].view(hidden_states.size(0), 1, 1)
            fused = fused + gate * adapter(hidden_states)
        return fused.to(source_dtype)


class RiskHead(nn.Module):
    def __init__(self, hidden_size, hidden_multiplier=0.5, dropout=0.1):
        super().__init__()
        inner = max(64, int(hidden_size * hidden_multiplier))
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, DIMENSION_COUNT),
        )

    def forward(self, pooled_hidden):
        target_dtype = self.net[1].weight.dtype
        pooled_hidden = pooled_hidden.to(target_dtype)
        return torch.sigmoid(self.net(pooled_hidden))


class RiskGatedCausalLM(nn.Module):
    def __init__(
        self,
        base_model_name_or_path,
        adapter_rank=8,
        adapter_alpha=16.0,
        adapter_dropout=0.05,
        dtype="auto",
    ):
        super().__init__()
        torch_dtype = None
        if dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif dtype == "fp16":
            torch_dtype = torch.float16

        self.base_model_name_or_path = base_model_name_or_path
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        hidden_size = self.base_model.config.hidden_size
        self.risk_head = RiskHead(hidden_size)
        self.adapter_bank = AdapterBank(hidden_size, rank=adapter_rank, alpha=adapter_alpha, dropout=adapter_dropout)

    @property
    def device(self):
        return next(self.parameters()).device

    def freeze_base(self):
        for param in self.base_model.parameters():
            param.requires_grad = False

    def freeze_risk_head(self):
        for param in self.risk_head.parameters():
            param.requires_grad = False

    def freeze_adapters(self):
        for param in self.adapter_bank.parameters():
            param.requires_grad = False

    def unfreeze_risk_head(self):
        for param in self.risk_head.parameters():
            param.requires_grad = True

    def unfreeze_adapters(self):
        for param in self.adapter_bank.parameters():
            param.requires_grad = True

    def lm_head(self, hidden_states):
        output_embeddings = self.base_model.get_output_embeddings()
        hidden_states = hidden_states.to(output_embeddings.weight.dtype)
        return output_embeddings(hidden_states)

    def base_requires_grad(self):
        return any(param.requires_grad for param in self.base_model.parameters())

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        use_cache=False,
    ):
        grad_context = nullcontext() if self.base_requires_grad() else torch.no_grad()
        with grad_context:
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
        hidden_states = outputs.hidden_states[-1]
        if not self.base_requires_grad():
            hidden_states = hidden_states.detach()
        pooled_hidden = hidden_states[:, -1, :]
        risk_scores = self.risk_head(pooled_hidden)
        fused_hidden = self.adapter_bank(hidden_states, risk_scores)
        logits = self.lm_head(fused_hidden)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            if shift_labels.ne(-100).any():
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
            else:
                loss = logits.new_zeros(())

        return {
            "loss": loss,
            "logits": logits,
            "risk_scores": risk_scores,
            "past_key_values": outputs.past_key_values,
        }

    @torch.no_grad()
    def score_messages(self, tokenizer, prompt_messages, completion_text, max_length=2048):
        from .data import render_chat

        was_training = self.training
        self.eval()
        full_messages = prompt_messages + [{"role": "assistant", "content": completion_text}]
        text = render_chat(tokenizer, full_messages, add_generation_prompt=False)
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.forward(encoded["input_ids"], attention_mask=encoded["attention_mask"])
        if was_training:
            self.train()
        return outputs["risk_scores"][0].detach().cpu()

    @torch.no_grad()
    def sample(
        self,
        input_ids,
        attention_mask,
        max_new_tokens=128,
        temperature=0.8,
        top_p=0.95,
        eos_token_id=None,
    ):
        was_training = self.training
        self.eval()
        generated = input_ids
        running_attention = attention_mask
        past_key_values = None
        current_input = input_ids
        finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)

        for _ in range(max_new_tokens):
            outputs = self.forward(
                current_input,
                attention_mask=running_attention,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs["logits"][:, -1, :]
            past_key_values = outputs["past_key_values"]

            if temperature <= 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                scaled = logits / temperature
                probs = torch.softmax(scaled, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cumulative = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumulative > top_p
                    mask[..., 1:] = mask[..., :-1].clone()
                    mask[..., 0] = False
                    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
                    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                    sample_idx = torch.multinomial(sorted_probs, num_samples=1)
                    next_token = torch.gather(sorted_indices, -1, sample_idx)
                else:
                    next_token = torch.multinomial(probs, num_samples=1)

            if eos_token_id is not None and finished.any():
                eos_fill = torch.full_like(next_token, eos_token_id)
                next_token = torch.where(finished.unsqueeze(-1), eos_fill, next_token)

            generated = torch.cat([generated, next_token], dim=1)
            running_attention = torch.cat(
                [running_attention, torch.ones_like(next_token, device=running_attention.device)],
                dim=1,
            )
            current_input = next_token

            if eos_token_id is not None:
                finished = finished | next_token.squeeze(-1).eq(eos_token_id)
                if torch.all(finished):
                    break

        if was_training:
            self.train()
        return generated, running_attention


def save_trainable_state(output_dir, model, extra_config=None):
    model = unwrap_model(model)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(model.risk_head.state_dict(), Path(output_dir) / "risk_head.pt")
    torch.save(model.adapter_bank.state_dict(), Path(output_dir) / "adapter_bank.pt")
    config = {
        "base_model_name_or_path": model.base_model_name_or_path,
        "dimension_count": DIMENSION_COUNT,
    }
    if extra_config:
        config.update(extra_config)
    with open(Path(output_dir) / "trainable_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def load_trainable_state(model, checkpoint_dir, strict=False):
    model = unwrap_model(model)
    risk_head_path = Path(checkpoint_dir) / "risk_head.pt"
    adapter_bank_path = Path(checkpoint_dir) / "adapter_bank.pt"
    if risk_head_path.exists():
        model.risk_head.load_state_dict(torch.load(risk_head_path, map_location="cpu"), strict=strict)
    if adapter_bank_path.exists():
        model.adapter_bank.load_state_dict(torch.load(adapter_bank_path, map_location="cpu"), strict=strict)


def sequence_logprob(logits, labels):
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target = labels[:, 1:].clone()
    valid_mask = target.ne(-100)
    target = target.masked_fill(~valid_mask, 0)
    token_log_probs = torch.gather(log_probs, dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs * valid_mask
    sequence_log_probs = token_log_probs.sum(dim=-1)
    token_counts = valid_mask.sum(dim=-1).clamp_min(1)
    return sequence_log_probs / token_counts
