from __future__ import annotations

import ast
import io
import keyword
import math
import re
import token as py_token
import tokenize as py_tokenize
from functools import lru_cache

def _check_randomness(program_str: str) -> bool:
    """尽量保守地检测随机性和时间依赖。"""
    pattern = (
        r"\brandom\b|\bnp\.random\b|\bnumpy\.random\b|"
        r"\b(randint|choice|choices|shuffle|sample|uniform|gauss)\s*\(|"
        r"\btime\b|\bdatetime\b|\b(perf_counter|monotonic|utcnow|now)\s*\("
    )
    return bool(re.search(pattern, program_str or "", re.IGNORECASE))


def _check_score_metadata_leak(program_str: str) -> bool:
    """检测代码是否显式读取了进化过程中的分数字段。"""
    text = str(program_str or "").lower()
    if not text:
        return False
    pattern = (
        r"\b(current_frontier_score|population_best_score|frontier_score|parent_scores?|parents_scores?)\b|"
        r"\bparent_score(?:_\d+)?\b|\bimprovement_over_(?:parent|parents|frontier)\b|\bscores?_after_addition\b"
    )
    return bool(re.search(pattern, text))


class BoundedQuadrantReward:
    """BQR v1：以 parent/frontier 为基线的四象限奖励。"""

    reward_type = "bqr_v1"
    positive_labels = {"bqr_q1_breakthrough", "bqr_q2_parent_tie_novelty"}
    structure_threshold = 0.50
    q2_reward_scale = 0.40
    q3_max_penalty = 0.05

    def __init__(
        self,
        *,
        minimize: bool = False,
        detect_randomness: bool = True,
        invalid_reward: float = -1.0,
        epsilon: float = 1.0e-6,
    ):
        self.minimize = bool(minimize)
        self.detect_randomness = bool(detect_randomness)
        self.invalid_reward = float(invalid_reward)
        self.epsilon = max(float(epsilon), 1.0e-12)

    def compute_bqr_result(
        self,
        *,
        program_str: str,
        score: float | None,
        eval_time: float | None,
        parent_codes: list[str] | tuple[str, ...] | None = None,
        parent_best_score: float | None = None,
        population_best_score: float | None = None,
    ) -> dict:
        del eval_time
        if score is None:
            return self.invalid_result("non_finite_score")
        score_f = float(score)
        if not math.isfinite(score_f):
            return self.invalid_result("non_finite_score")
        if self.detect_randomness and _check_randomness(program_str):
            return self.invalid_result("randomness_detected")
        if _check_score_metadata_leak(program_str):
            return self.invalid_result("metadata_leak")

        parent_best = self._finite_value(parent_best_score)
        frontier_best = self._finite_value(population_best_score)
        frontier_best = frontier_best if frontier_best is not None else parent_best
        baseline = parent_best if parent_best is not None else frontier_best
        if baseline is None or frontier_best is None:
            return self.invalid_result("missing_baseline")

        utility = self._utility(score_f)
        frontier_utility = self._utility(frontier_best)
        parent_utility = self._utility(parent_best) if parent_best is not None else None
        baseline_utility = self._utility(baseline)
        signed_delta = utility - baseline_utility
        relative_delta = signed_delta / max(min(abs(utility), abs(baseline_utility)), self.epsilon)
        delta_frontier = utility - frontier_utility
        delta_frontier_rel = delta_frontier / max(abs(frontier_utility), 1.0)
        delta_parent = utility - parent_utility if parent_utility is not None else None
        delta_parent_rel = delta_parent / max(abs(parent_utility), 1.0) if delta_parent is not None and parent_utility is not None else None
        beats_frontier = delta_frontier > self.epsilon
        beats_parent = bool(delta_parent is not None and delta_parent > self.epsilon)
        ties_parent = bool(delta_parent is not None and not beats_parent and abs(delta_parent) <= self.epsilon)

        parent_code_list = self._clean_code_list(parent_codes)
        candidate_code = self._primary_function_source(program_str)
        structure_novelty = self._prompt_parent_structure_diversity(candidate_code, parent_code_list)
        q2_enabled = bool(parent_code_list and self._structure_tokens(candidate_code))
        structure_passes_q2 = bool(q2_enabled and structure_novelty is not None and structure_novelty + self.epsilon >= self.structure_threshold)

        common = {
            "score": score_f,
            "parent_best_score": parent_best_score,
            "population_best_score": population_best_score,
            "baseline_used": baseline,
            "signed_delta": signed_delta,
            "relative_delta": relative_delta,
            "delta_parent": delta_parent,
            "delta_parent_rel": delta_parent_rel,
            "delta_frontier": delta_frontier,
            "delta_frontier_rel": delta_frontier_rel,
            "structure_novelty": structure_novelty,
            "structural_threshold": self.structure_threshold,
            "q2_enabled": q2_enabled,
            "structure_passes_q2": structure_passes_q2,
            "failure_label": None,
            "beats_parent": beats_parent,
            "ties_parent": ties_parent,
            "beats_frontier": beats_frontier,
            "performance_improved": bool(beats_frontier or beats_parent),
        }

        if beats_parent or (parent_utility is None and beats_frontier):
            q1_delta = delta_parent_rel if delta_parent_rel is not None else delta_frontier_rel
            reward = 1.0 + max(0.0, min(1.0, q1_delta if q1_delta is not None else 0.0))
            return {**common, "reward": float(max(1.0, min(2.0, reward))), "reward_label": "bqr_q1_breakthrough", "reward_quadrant": "q1"}

        if q2_enabled and ties_parent and structure_passes_q2:
            reward = self.q2_reward_scale * max(0.0, min(1.0, structure_novelty or 0.0))
            return {**common, "reward": float(max(0.0, min(self.q2_reward_scale, reward))), "reward_label": "bqr_q2_parent_tie_novelty", "reward_quadrant": "q2"}

        diversity = max(0.0, min(1.0, structure_novelty or 0.0))
        reward = -self.q3_max_penalty * (1.0 - min(diversity / self.structure_threshold, 1.0))
        return {**common, "reward": float(reward), "reward_label": "bqr_q3_homogeneity", "reward_quadrant": "q3"}

    def compute_reward(self, program_str: str, score: float | None, eval_time: float | None, **kwargs) -> float:
        if kwargs.get("validity") is not None and not bool(kwargs.get("validity")):
            return self.invalid()
        if kwargs.get("failure_level") or score is None:
            return self.invalid()
        score_f = self._finite_value(score)
        if score_f is None:
            return self.invalid()
        return float(
            self.compute_bqr_result(
                program_str=program_str,
                score=score_f,
                eval_time=eval_time,
                parent_codes=kwargs.get("parent_codes"),
                parent_best_score=kwargs.get("parent_best_score"),
                population_best_score=kwargs.get("population_best_score", kwargs.get("best_score")),
            )["reward"]
        )

    def compute_batch(self, batch: list[dict]) -> list[float]:
        return [self.compute_reward(**item) for item in batch]

    def _utility(self, score: float) -> float:
        return -float(score) if self.minimize else float(score)

    @staticmethod
    def _finite_value(value: float | None) -> float | None:
        try:
            if value is None:
                return None
            parsed = float(value)
            return parsed if math.isfinite(parsed) else None
        except (TypeError, ValueError):
            return None

    @classmethod
    def _clean_code_list(cls, codes: list[str] | tuple[str, ...] | None) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for code in codes or []:
            text = cls._primary_function_source(code)
            if not text or text in seen:
                continue
            seen.add(text)
            cleaned.append(text)
        return cleaned

    @classmethod
    def _primary_function_source(cls, code: str | None) -> str:
        text = str(code or "").strip()
        if not text:
            return ""
        try:
            tree = ast.parse(text)
        except Exception:
            return text
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                segment = ast.get_source_segment(text, node)
                if segment:
                    return segment.strip()
                return cls._function_stub(node)
        return text

    @staticmethod
    def _function_stub(node: ast.FunctionDef) -> str:
        return " ".join(["def " + node.name + "(...):", *(type(child).__name__ for child in ast.walk(node))])

    @classmethod
    def structure_diversity(cls, parent_code: str | None, candidate_code: str | None) -> float:
        parent_tokens = cls._structure_tokens(parent_code)
        candidate_tokens = cls._structure_tokens(candidate_code)
        return 0.0 if not candidate_tokens else len(candidate_tokens - parent_tokens) / len(candidate_tokens)

    @classmethod
    def symmetric_structure_diversity(cls, left_code: str | None, right_code: str | None) -> float:
        left_tokens = cls._structure_tokens(left_code)
        right_tokens = cls._structure_tokens(right_code)
        union = left_tokens | right_tokens
        return 0.0 if not union else 1.0 - (len(left_tokens & right_tokens) / len(union))

    @classmethod
    def _prompt_parent_structure_diversity(
        cls,
        candidate_code: str,
        parent_codes: list[str] | tuple[str, ...] | None,
    ) -> float | None:
        parent_code_list = [cls._primary_function_source(code) for code in parent_codes or []]
        parent_code_list = [code for code in parent_code_list if code]
        if not parent_code_list:
            return None
        return min(cls.structure_diversity(parent_code, candidate_code) for parent_code in parent_code_list)

    @classmethod
    @lru_cache(maxsize=20000)
    def _code_component_tokens(cls, code: str | None) -> frozenset[str]:
        text = cls._primary_function_source(code)
        if not text:
            return frozenset()
        params = cls._signature_param_names(text)
        allowed_ops = {"+", "-", "*", "/", "**", "%", "<", "<=", ">", ">=", "==", "!="}
        try:
            raw_tokens = [
                tok
                for tok in py_tokenize.generate_tokens(io.StringIO(text).readline)
                if tok.type
                not in {
                    py_tokenize.NL,
                    py_tokenize.NEWLINE,
                    py_tokenize.INDENT,
                    py_tokenize.DEDENT,
                    py_tokenize.ENDMARKER,
                    py_tokenize.COMMENT,
                }
            ]
        except Exception:
            return frozenset(re.findall(r"[A-Za-z_][A-Za-z0-9_]*|==|!=|<=|>=|//|\*\*|[+\-*/%<>=]", text))
        tokens: set[str] = set()
        meaningful = [tok for tok in raw_tokens if tok.type != py_token.STRING]
        for idx, tok in enumerate(meaningful):
            prev_text = meaningful[idx - 1].string if idx > 0 else ""
            next_text = meaningful[idx + 1].string if idx + 1 < len(meaningful) else ""
            if tok.type == py_token.NAME:
                name = tok.string
                if keyword.iskeyword(name):
                    tokens.add(f"KW:{name}")
                elif name in params:
                    tokens.add(f"ARG:{name}")
                elif prev_text == ".":
                    tokens.add(f"{'CALL' if next_text == '(' else 'ATTR'}:{name}")
                elif next_text == "(" and prev_text not in {"def", "class"}:
                    tokens.add(f"CALL:{name}")
            elif tok.type == py_token.OP and tok.string in allowed_ops:
                tokens.add(f"OP:{tok.string}")
            elif tok.type == py_token.NUMBER:
                bucket = cls._number_bucket(tok.string)
                if bucket:
                    tokens.add(bucket)
        return frozenset(tokens)

    @staticmethod
    def _signature_param_names(text: str) -> tuple[str, ...]:
        try:
            tree = ast.parse(text)
        except Exception:
            return ()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                args = [*(arg.arg for arg in node.args.posonlyargs), *(arg.arg for arg in node.args.args), *(arg.arg for arg in node.args.kwonlyargs)]
                return tuple(args)
        return ()

    @staticmethod
    def _number_bucket(text: str) -> str:
        try:
            value = float(text)
        except Exception:
            return ""
        if value == 0.0:
            return "NUM_ZERO"
        if value == 1.0:
            return "NUM_ONE"
        if value > 0.0 and abs(value) < 1.0:
            return "NUM_SMALLFLOAT"
        return "NUM_POS" if value > 0.0 else "NUM_NEG"

    @classmethod
    def _structure_tokens(cls, code: str | None) -> frozenset[str]:
        return cls._code_component_tokens(code)

    def invalid_result(self, failure_label: str) -> dict:
        failure = str(failure_label or "invalid")
        return {
            "reward": float(self.invalid_reward),
            "reward_label": "bqr_q4_invalid",
            "reward_quadrant": "q4",
            "failure_label": failure,
            "score": None,
            "parent_best_score": None,
            "population_best_score": None,
            "baseline_used": None,
            "signed_delta": None,
            "relative_delta": None,
            "delta_parent": None,
            "delta_parent_rel": None,
            "delta_frontier": None,
            "delta_frontier_rel": None,
            "structure_novelty": None,
            "structural_threshold": None,
            "q2_enabled": False,
            "structure_passes_q2": False,
            "beats_parent": False,
            "ties_parent": False,
            "beats_frontier": False,
            "performance_improved": False,
        }

    def failure_reward_label(self, failure_label: str) -> str:
        return str(self.invalid_result(failure_label).get("reward_label") or "invalid")

    def invalid(self) -> float:
        return self.invalid_reward

    def get_state(self) -> dict:
        return {}

    def load_state(self, state: dict):
        del state
        return None


def build_reward_fn_from_task_rl(task_rl: dict, logger=None):
    reward_type = str(task_rl.get("reward_type", "bqr_v1")).strip().lower()
    if reward_type != "bqr_v1":
        raise ValueError(f"unknown reward_type: {reward_type!r}, only 'bqr_v1' is supported")
    reward_fn = BoundedQuadrantReward(
        minimize=task_rl.get("minimize", False),
        detect_randomness=task_rl.get("detect_randomness", True),
        invalid_reward=task_rl.get("invalid_reward", -1.0),
        epsilon=task_rl.get("epsilon", 1.0e-6),
    )
    if logger is not None:
        logger.info(
            "奖励函数: BQR v1 parent-relative (minimize=%s, invalid_reward=%.1f, epsilon=%.1e, q2_tau=%.2f, q3_max_penalty=%.2f)",
            task_rl.get("minimize", False),
            task_rl.get("invalid_reward", -1.0),
            task_rl.get("epsilon", 1.0e-6),
            BoundedQuadrantReward.structure_threshold,
            BoundedQuadrantReward.q3_max_penalty,
        )
    return reward_fn
