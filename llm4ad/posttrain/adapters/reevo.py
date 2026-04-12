from __future__ import annotations

from .evolution import EvolutionAdapter


class ReEvoAdapter(EvolutionAdapter):
    def record_short_reflection(self, *, prompt: str, response: str) -> None:
        self.after_sample(prompt=prompt, response=response)

    def record_long_reflection(self, *, prompt: str, response: str) -> None:
        self.after_sample(prompt=prompt, response=response)
