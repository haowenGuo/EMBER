"""OpenAI-compatible provider adapter."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class OpenAICompatibleProvider:
    """OpenAI-compatible chat completions provider.

    Environment defaults:
    - ``OPENAI_API_KEY``
    - ``OPENAI_BASE_URL``
    - ``OPENAI_MODEL``
    """

    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.2
    max_tokens: int = 1024

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError("Install the llm extra first: pip install -e .[llm]") from exc

        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the openai provider.")
        model = self.model or os.getenv("OPENAI_MODEL")
        if not model:
            raise RuntimeError("OPENAI_MODEL or --model is required for the openai provider.")

        client = OpenAI(
            api_key=api_key,
            base_url=self.base_url or os.getenv("OPENAI_BASE_URL") or None,
        )
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""
