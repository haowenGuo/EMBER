"""Base provider interfaces."""

from __future__ import annotations

from typing import Protocol


class TextProvider(Protocol):
    """Provider that turns a prompt into text."""

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        ...
