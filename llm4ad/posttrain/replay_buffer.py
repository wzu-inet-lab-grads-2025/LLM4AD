from __future__ import annotations

from collections import deque

from .schemas import SampleRecord


def _scalar_score(score):
    if isinstance(score, (int, float)):
        return float(score)
    return None


class ReplayBuffer:
    def __init__(self, config):
        self._config = config
        self._recent = deque(maxlen=config.recent_capacity)
        self._elite: list[SampleRecord] = []
        self._diversity: list[SampleRecord] = []
        self._diversity_keys: set[tuple[str | None, str | None]] = set()

    def add_records(self, records: list[SampleRecord]) -> None:
        for record in records:
            self._recent.append(record)
            self._maybe_add_elite(record)
            self._maybe_add_diversity(record)

    def sample_recent(self, n: int) -> list[SampleRecord]:
        if n <= 0:
            return []
        return list(self._recent)[-n:]

    def sample_elite(self, n: int) -> list[SampleRecord]:
        if n <= 0:
            return []
        return self._elite[:n]

    def sample_diversity(self, n: int) -> list[SampleRecord]:
        if n <= 0:
            return []
        return self._diversity[:n]

    def _maybe_add_elite(self, record: SampleRecord) -> None:
        scalar = _scalar_score(record.score)
        if scalar is None:
            return
        self._elite.append(record)
        self._elite.sort(key=lambda item: _scalar_score(item.score), reverse=True)
        del self._elite[self._config.elite_capacity :]

    def _maybe_add_diversity(self, record: SampleRecord) -> None:
        key = (record.function, record.algorithm)
        if key in self._diversity_keys:
            return
        self._diversity_keys.add(key)
        self._diversity.append(record)
        if len(self._diversity) > self._config.diversity_capacity:
            removed = self._diversity.pop(0)
            self._diversity_keys.discard((removed.function, removed.algorithm))
