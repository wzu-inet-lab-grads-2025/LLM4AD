from __future__ import annotations

import math
from typing import Iterable, List

import numpy as np

from ...base import Function


class Population:
    """EoH-style two-zone population.

    `_population` is the active parent pool. New candidates are accumulated in
    `_next_gen_pop` and only affect parent selection after `survival()`.
    """

    def __init__(
        self,
        pop_size: int,
        generation: int = 0,
        pop: List[Function] | "Population" | None = None,
        *,
        next_gen_pop: List[Function] | None = None,
        minimize: bool = False,
    ):
        self._pop_size = int(pop_size)
        self._generation = int(generation)
        self._minimize = bool(minimize)
        self._population = list(pop._population if isinstance(pop, Population) else (pop or []))
        self._next_gen_pop = list(next_gen_pop or [])
        self._population = self._deduplicate(self._population)
        self._sort_in_place(self._population)
        self._population = self._population[: self._pop_size]

    def __len__(self) -> int:
        return len(self._population)

    def __getitem__(self, item) -> Function:
        return self._population[item]

    @property
    def population(self) -> list[Function]:
        return self._population

    @property
    def next_generation(self) -> list[Function]:
        return self._next_gen_pop

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def next_generation_size(self) -> int:
        return len(self._next_gen_pop)

    def add_initial_member(self, func: Function) -> tuple[bool, bool]:
        if not self._has_finite_score(func) or self.has_duplicate_function(func):
            return False, False
        if len(self._population) >= self._pop_size:
            return False, False
        self._population.append(func)
        self._sort_in_place(self._population)
        return True, len(self._population) >= self._pop_size

    def register_evolved_function(self, func: Function) -> tuple[bool, bool, list[Function]]:
        if not self._has_finite_score(func) or self.has_duplicate_function(func):
            return False, False, []
        self._next_gen_pop.append(func)
        if len(self._next_gen_pop) < self._pop_size:
            return True, False, []
        evicted = self.survival()
        return True, True, evicted

    def survival(self) -> list[Function]:
        before = list(self._population)
        combined = self._deduplicate([*self._population, *self._next_gen_pop])
        self._sort_in_place(combined)
        self._population = combined[: self._pop_size]
        self._next_gen_pop = []
        self._generation += 1
        return [old for old in before if all(old is not cur for cur in self._population)]

    def replace_population(self, members: Iterable[Function]) -> None:
        self._population = self._deduplicate([func for func in members if self._has_finite_score(func)])
        self._sort_in_place(self._population)
        self._population = self._population[: self._pop_size]
        self._next_gen_pop = []

    def has_duplicate_function(self, func: str | Function) -> bool:
        return any(self._same_member(existing, func) for existing in [*self._population, *self._next_gen_pop])

    def selection(self) -> Function:
        funcs = [func for func in self._population if self._has_finite_score(func)]
        if not funcs:
            raise RuntimeError("EoH-RL parent selection requires a non-empty active population")
        self._sort_in_place(funcs)
        n = len(funcs)
        probs = np.empty(n, dtype=float)
        rank = 0
        while rank < n:
            end = rank + 1
            score = self._utility(funcs[rank].score)
            while end < n and self._utility(funcs[end].score) == score:
                end += 1
            probs[rank:end] = sum(1.0 / (r + n) for r in range(rank, end)) / (end - rank)
            rank = end
        probs = probs / probs.sum()
        return np.random.choice(funcs, p=probs)

    def _deduplicate(self, funcs: list[Function]) -> list[Function]:
        unique: list[Function] = []
        for func in funcs:
            if not self._has_finite_score(func):
                continue
            if any(self._same_member(existing, func) for existing in unique):
                continue
            unique.append(func)
        return unique

    def _sort_in_place(self, funcs: list[Function]) -> None:
        funcs.sort(key=lambda func: self._utility(func.score), reverse=True)

    def _utility(self, score) -> float:
        return -float(score) if self._minimize else float(score)

    @staticmethod
    def _same_member(left: str | Function, right: str | Function) -> bool:
        return Population._code_key(left) == Population._code_key(right)

    @staticmethod
    def _code_key(func: str | Function) -> str:
        text = str(func or "").strip()
        return "\n".join(line.rstrip() for line in text.splitlines()).strip()

    @staticmethod
    def _has_finite_score(func: Function) -> bool:
        score = getattr(func, "score", None)
        try:
            return score is not None and math.isfinite(float(score))
        except Exception:
            return False
