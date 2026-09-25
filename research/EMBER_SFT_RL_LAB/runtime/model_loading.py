import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def _is_adapter_directory(model_name_or_path):
    return os.path.isdir(model_name_or_path) and os.path.exists(
        os.path.join(model_name_or_path, "adapter_config.json")
    )


def _load_adapter_tokenizer(adapter_path):
    config_path = os.path.join(adapter_path, "adapter_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        adapter_config = json.load(f)

    base_model_name = adapter_config.get("base_model_name_or_path") or adapter_path
    return AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)


def load_local_text_generation_pipeline(
    model_name_or_path,
    device="cuda:0",
    max_new_tokens=1024,
    do_sample=False,
    temperature=0.1,
    torch_dtype=torch.float16,
):
    if _is_adapter_directory(model_name_or_path):
        try:
            from peft import AutoPeftModelForCausalLM
        except ImportError as exc:
            raise ImportError(
                "Detected a PEFT adapter directory but the 'peft' package is not installed. "
                "Install it with: pip install peft"
            ) from exc

        tokenizer = _load_adapter_tokenizer(model_name_or_path)
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        if device and device != "auto":
            model = model.to(device)

        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )

    return pipeline(
        "text-generation",
        max_new_tokens=max_new_tokens,
        model=model_name_or_path,
        device_map=device,
        torch_dtype=torch_dtype,
        do_sample=do_sample,
        temperature=temperature,
    )
