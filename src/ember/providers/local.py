"""Local model provider adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TransformersProvider:
    """Local Transformers text-generation provider."""

    model: str
    device_map: str = "auto"
    max_new_tokens: int = 512
    temperature: float = 0.2
    _pipe: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError("Install the local extra first: pip install -e .[local]") from exc

        self._pipe = pipeline(
            "text-generation",
            model=self.model,
            device_map=self.device_map,
            torch_dtype=torch.float16,
        )

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        text = f"{system}\n\n{prompt}" if system else prompt
        output = self._pipe(
            text,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
        )
        generated = output[0]["generated_text"]
        return generated[len(text) :].strip() if generated.startswith(text) else generated
