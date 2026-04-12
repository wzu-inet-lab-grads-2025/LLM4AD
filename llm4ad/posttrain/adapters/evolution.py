from __future__ import annotations

from .base import MethodAdapterBase


class EvolutionAdapter(MethodAdapterBase):
    def __init__(self, *, round_config=None, event_store=None):
        super().__init__(round_config=round_config, event_store=event_store)
        self._round_start_sample_num = 0
        self._round_start_generation = 0

    def bind(self, method, runtime, event_store=None) -> None:
        super().bind(method, runtime, event_store=event_store)
        self._round_start_sample_num = getattr(method, "_tot_sample_nums", 0)
        population = getattr(method, "_population", None)
        self._round_start_generation = getattr(population, "generation", 0)

    def should_stop_round(self, method) -> bool:
        if self._round_config is None:
            return False

        max_samples = getattr(self._round_config, "max_samples_per_round", None)
        max_generations = getattr(self._round_config, "max_generations_per_round", None)

        if max_samples is None and max_generations is None:
            return False

        current_samples = getattr(method, "_tot_sample_nums", 0)
        delta_samples = current_samples - self._round_start_sample_num
        if max_samples is not None and delta_samples >= max_samples:
            return True

        population = getattr(method, "_population", None)
        current_generation = getattr(population, "generation", 0)
        delta_generations = current_generation - self._round_start_generation
        if max_generations is not None and delta_generations >= max_generations:
            return True

        return False

    def snapshot_state(self, method) -> dict:
        population = getattr(method, "_population", None)
        return {
            "round_start_sample_num": self._round_start_sample_num,
            "round_start_generation": self._round_start_generation,
            "current_generation": getattr(population, "generation", None),
        }
