from __future__ import annotations

import ast
import math
from typing import Iterable, List

import numpy as np

from ...base import Function


def _ast_signature(code) -> dict[str, float]:
    try:
        tree = ast.parse(str(code))
    except Exception:
        return {}
    sig: dict[str, float] = {}

    def add(key: str, weight: float = 1.0):
        sig[key] = sig.get(key, 0.0) + weight

    def call_name(node) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = call_name(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        return ""

    for parent in ast.walk(tree):
        add(type(parent).__name__)
        if isinstance(parent, ast.Call):
            name = call_name(parent.func)
            if name:
                add(f"call:{name}", 0.5)
        for child in ast.iter_child_nodes(parent):
            add(f"{type(parent).__name__}->{type(child).__name__}", 0.5)
    return sig


def ast_distance(left, right) -> float:
    a, b = _ast_signature(left), _ast_signature(right)
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    norm = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return 0.0 if norm == 0.0 else max(0.0, min(1.0, 1.0 - dot / norm))


class Population:
    """CALM-style single population.

    `_population` is the full archive of valid, code-unique algorithms. Parent
    selection uses only the top `pop_size` members after score sorting.
    """

    def __init__(
        self,
        pop_size: int,
        generation: int = 0,
        pop: List[Function] | "Population" | None = None,
        *,
        minimize: bool = False,
        seed: int = 0,
    ):
        self._pop_size = int(pop_size)
        self._generation = int(generation)
        self._minimize = bool(minimize)
        self._rng = np.random.default_rng(int(seed))
        self._population = list(pop._population if isinstance(pop, Population) else (pop or []))
        self._population = self._deduplicate(self._population)
        self._sort_in_place(self._population)

    def __len__(self) -> int:
        return len(self.active_population)

    def __getitem__(self, item) -> Function:
        return self.active_population[item]

    @property
    def population(self) -> list[Function]:
        return self._population

    @property
    def active_population(self) -> list[Function]:
        return self._population[: self._pop_size]

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def archive_size(self) -> int:
        return len(self._population)

    def add_initial_member(self, func: Function) -> tuple[bool, bool]:
        if not self._add(func):
            return False, False
        return True, len(self.active_population) >= self._pop_size

    def register_evolved_function(self, func: Function) -> tuple[bool, bool, list[Function]]:
        before = list(self.active_population)
        if not self._add(func):
            return False, False, []
        self._generation += 1
        active = self.active_population
        survived = any(func is cur for cur in active)
        evicted = [old for old in before if all(old is not cur for cur in active)]
        return True, survived, evicted

    def replace_population(self, members: Iterable[Function]) -> None:
        self._population = self._deduplicate([func for func in members if self._has_finite_score(func)])
        self._sort_in_place(self._population)

    def has_duplicate_function(self, func: str | Function) -> bool:
        return any(self._same_member(existing, func) for existing in self._population)

    def selection(self) -> Function:
        funcs = [func for func in self.active_population if self._has_finite_score(func)]
        if not funcs:
            raise RuntimeError("EoH-RL parent selection requires a non-empty active population")
        rank = 1 + np.arange(len(funcs), dtype=float)
        probs = (1.0 / rank) / np.sum(1.0 / rank)
        return self._rng.choice(funcs, p=probs)

    def crossover_selection(self) -> list[Function]:
        parent1 = self.active_population[0]
        candidates = [func for func in self._population if func is not parent1 and self._has_finite_score(func)]
        if not candidates:
            raise RuntimeError("EoH-RL crossover requires two distinct parents")
        parent2 = max(candidates, key=lambda func: ast_distance(parent1, func))
        return [parent1, parent2]

    def _add(self, func: Function) -> bool:
        if not self._has_finite_score(func) or self.has_duplicate_function(func):
            return False
        self._population.append(func)
        self._sort_in_place(self._population)
        return True

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
