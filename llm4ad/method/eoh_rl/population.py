from __future__ import annotations

import math
from typing import List

from ...base import Function


class Population:
    def __init__(
        self,
        pop_size,
        generation=0,
        pop: List[Function] | Population | None = None,
        *,
        minimize: bool = False,
    ):
        if pop is None:
            self._population = []
        elif isinstance(pop, list):
            self._population = pop
        else:
            self._population = pop._population

        self._pop_size = pop_size
        self._generation = generation
        self._minimize = bool(minimize)
        self._sort_population()
        self._truncate_population()

    def __len__(self):
        return len(self._population)

    def __getitem__(self, item) -> Function:
        return self._population[item]

    def __setitem__(self, key, value):
        self._population[key] = value

    @property
    def population(self):
        return self._population

    @property
    def generation(self):
        return self._generation

    def add_initial_member(self, func: Function) -> tuple[bool, bool]:
        if not self._has_finite_score(func):
            return False, False
        target = int(self._pop_size or 0)
        if target > 0 and len(self._population) >= target:
            return False, False
        if self.has_duplicate_function(func):
            return False, False
        self._population.append(func)
        self._sort_population()
        just_filled = target > 0 and len(self._population) >= target
        return True, just_filled

    def register_evolved_function(self, func: Function) -> tuple[bool, bool, list[Function]]:
        if not self._has_finite_score(func):
            return False, False, []
        target = int(self._pop_size or 0)
        was_full = target > 0 and len(self._population) >= target
        if self.has_duplicate_function(func):
            return False, False, []
        before = list(self._population)
        self._population.append(func)
        self._sort_population()
        self._truncate_population()
        inserted = any(existing is func for existing in self._population)
        just_filled = (not was_full) and target > 0 and len(self._population) >= target
        evicted = [old for old in before if all(old is not current for current in self._population)]
        return inserted, just_filled, evicted if inserted else []

    def replace_population(self, members: List[Function]) -> None:
        cleaned = []
        for func in members or []:
            if not self._has_finite_score(func):
                continue
            if any(self._same_member(existing, func) for existing in cleaned):
                continue
            cleaned.append(func)
        self._population = cleaned
        self._sort_population()
        self._truncate_population()

    def increment_generation(self) -> None:
        self._generation += 1

    def has_duplicate_function(self, func: str | Function) -> bool:
        for existing in self._population:
            if self._same_member(existing, func):
                return True
        return False

    def _sort_population(self) -> None:
        self._population.sort(key=lambda func: float(func.score), reverse=not self._minimize)

    def _truncate_population(self) -> None:
        target = int(self._pop_size or 0)
        if target > 0 and len(self._population) > target:
            self._population = self._population[:target]

    @staticmethod
    def _same_member(left: str | Function, right: str | Function) -> bool:
        return str(left).strip() == str(right).strip()

    @staticmethod
    def _has_finite_score(func: Function) -> bool:
        score = getattr(func, "score", None)
        if score is None:
            return False
        try:
            return math.isfinite(float(score))
        except Exception:
            return False
