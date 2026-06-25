"""Multi-agent adversarial dialogue arena used by EMBER."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable


AgentFn = Callable[[list[dict[str, str]]], str]


@dataclass(slots=True)
class DialogueTurn:
    sender: str
    content: str
    round_index: int


class MultiAgentDebateArena:
    """Simple debate arena for dynamic bias evaluation.

    The target model is repeatedly challenged by one or more provocateur
    agents. The arena records a full transcript so downstream evaluators can
    score bias by round, model, scenario, or dimension.
    """

    def __init__(
        self,
        target_name: str,
        target_agent: AgentFn,
        provocateurs: dict[str, AgentFn],
        topic: str,
    ) -> None:
        self.target_name = target_name
        self.target_agent = target_agent
        self.provocateurs = provocateurs
        self.topic = topic
        self.turns: list[DialogueTurn] = [
            DialogueTurn("referee", f"Debate topic: {topic}", 0)
        ]

    def run_round(self, round_index: int) -> list[DialogueTurn]:
        history = self.as_messages()
        if round_index == 0:
            target_response = self.target_agent(history)
            self.turns.append(DialogueTurn(self.target_name, target_response, 0))
            return self.turns

        for name, agent in self.provocateurs.items():
            attack = agent(self.as_messages())
            self.turns.append(DialogueTurn(name, attack, round_index))

        target_response = self.target_agent(self.as_messages())
        self.turns.append(DialogueTurn(self.target_name, target_response, round_index))
        return self.turns

    def run(self, rounds: Iterable[int]) -> list[DialogueTurn]:
        for round_index in rounds:
            self.run_round(round_index)
        return self.turns

    def as_messages(self) -> list[dict[str, str]]:
        return [
            {"role": "assistant" if turn.sender == self.target_name else "user", "content": turn.content}
            for turn in self.turns
        ]

    def target_responses(self) -> list[str]:
        return [turn.content for turn in self.turns if turn.sender == self.target_name]

    def transcript(self) -> str:
        return "\n".join(
            f"[round {turn.round_index}] {turn.sender}: {turn.content}"
            for turn in self.turns
        )
