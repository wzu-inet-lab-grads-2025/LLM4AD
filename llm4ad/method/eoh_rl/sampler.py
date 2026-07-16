from __future__ import annotations

import ast
import logging
import re
from typing import Optional

from ...base import Function, LLM, Program, SampleTrimmer, TextFunctionProgramConverter
from .rl.sft_data import validate_generated_code

logger = logging.getLogger(__name__)


class EoHSampler:
    def __init__(
        self,
        llm: Optional[LLM],
        template_program: str | Program,
        *,
        enable_ast_gate: bool = False,
    ):
        self.llm = llm
        self._template_program = template_program
        self._template_program_obj = (
            TextFunctionProgramConverter.text_to_program(template_program)
            if isinstance(template_program, str)
            else template_program
        )
        self._template_function = self._template_program_obj.functions[0] if self._template_program_obj else None
        self._enable_ast_gate = bool(enable_ast_gate)

    def get_strategy_and_function_records_batch(self, prompt: str, n: int) -> list[dict]:
        if n <= 0:
            return []
        responses = self._draw_batch_responses(prompt, n)
        records: list[dict] = []
        for response in responses[:n]:
            parsed = self.parse_response_record(response)
            parsed["raw_response"] = response
            records.append(parsed)
        return records

    def _draw_batch_responses(self, prompt: str, n: int) -> list[str]:
        if n <= 0:
            return []
        if self.llm is None or not getattr(self.llm, "has_batch_draw", False):
            return [self._draw_valid_response(prompt) for _ in range(n)]

        responses: list[str] = []
        remaining = int(n)
        max_rounds = 3
        rounds = 0
        while remaining > 0 and rounds < max_rounds:
            rounds += 1
            try:
                batch = self.llm.draw_samples(prompt, remaining)
            except Exception:
                batch = []
            if not isinstance(batch, (list, tuple)) or not batch:
                break
            for item in batch:
                response = str(item or "")
                if not self._ast_gate_check(response):
                    continue
                responses.append(response)
                if len(responses) >= n:
                    break
            remaining = n - len(responses)
            if remaining > 0 and rounds < max_rounds:
                logger.info(
                    "[EoHSampler] batch AST gate kept %d/%d samples; drawing another batch (%d/%d)",
                    len(responses),
                    n,
                    rounds + 1,
                    max_rounds,
                )
        if len(responses) < n:
            responses.extend(self._draw_valid_response(prompt) for _ in range(n - len(responses)))
        return responses[:n]

    def _draw_valid_response(self, prompt: str) -> str:
        if self.llm is None:
            raise RuntimeError("EoHSampler requires a live llm to draw responses")
        response = ""
        for attempt_idx in range(3):
            response = self.llm.draw_sample(prompt)
            if self._ast_gate_check(response):
                return response
            logger.info("[EoHSampler] AST gate rejected sample; retrying (%d/%d)", attempt_idx + 1, 3)
        return response

    def _ast_gate_check(self, response: str) -> bool:
        return self._ast_gate_failure_reason(response) is None

    def _ast_gate_failure_reason(self, response: str) -> str | None:
        if not self._enable_ast_gate:
            return None
        python = self.extract_python_from_response(response)
        result = validate_generated_code(python, template_program=str(self._template_program))
        return None if result.ok else str(result.reason or "ast_gate_rejected")

    def parse_response_record(self, response: str) -> dict:
        text = str(response or "").strip()
        contract = self._contract_flags(text)
        if not text:
            return {**self._failure("no_output"), **contract}

        strategy = self.trim_strategy_from_response(text)
        if strategy and self._strategy_has_metadata_leak(strategy):
            return {**self._failure("metadata_leak", strategy=strategy), **contract}

        python = self.extract_python_from_response(text)
        if not python:
            return {**self._failure("missing_function", strategy=strategy), **contract}

        return {**self._parse_record(strategy, python), **contract}

    @classmethod
    def trim_strategy_from_response(cls, response: str) -> str | None:
        return cls._extract_braced_idea(response)

    def _parse_record(self, strategy: str | None, python: str) -> dict:
        try:
            tree = ast.parse(python)
        except SyntaxError:
            return self._failure("syntax_error", strategy=strategy, python=python)
        except Exception:
            return self._failure("parse_error", strategy=strategy, python=python)

        functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        disallowed_top_level = [
            node
            for node in tree.body
            if not isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        if len(functions) != 1 or disallowed_top_level or not isinstance(functions[0], ast.FunctionDef):
            if len(functions) > 1:
                return self._failure("multiple_functions", strategy=strategy, python=python)
            return self._failure("missing_function", strategy=strategy, python=python)

        function_node = functions[0]
        if function_node.decorator_list:
            return self._failure("parse_error", strategy=strategy, python=python)
        if any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Global, ast.Nonlocal))
            for node in ast.walk(function_node)
            if node is not function_node
        ):
            return self._failure("parse_error", strategy=strategy, python=python)

        try:
            func = TextFunctionProgramConverter.text_to_function(python)
        except ValueError:
            return self._failure("multiple_functions", strategy=strategy, python=python)
        except Exception:
            return self._failure("parse_error", strategy=strategy, python=python)
        if func is None:
            return self._failure("parse_error", strategy=strategy, python=python)
        setattr(func, "_eoh_python", python)
        if not self._matches_template_function(func):
            template_func = self._template_function
            failure = "wrong_function_name" if template_func is not None and func.name != template_func.name else "wrong_signature"
            return self._failure(failure, strategy=strategy, python=python)
        program = TextFunctionProgramConverter.function_to_program(func, self._template_program_obj)
        if program is None:
            return self._failure("parse_error", strategy=strategy, python=python)
        return {
            "failure_label": None,
            "strategy": strategy,
            "python": python,
            "func": func,
            "program": program,
        }

    def _matches_template_function(self, func: Function | None) -> bool:
        if func is None or self._template_function is None:
            return False
        if func.name != self._template_function.name:
            return False
        if func.args != self._template_function.args:
            return False
        return (func.return_type or "") == (self._template_function.return_type or "")

    @staticmethod
    def _extract_braced_idea(text: str) -> str | None:
        raw = re.split(r"```|^\s*def\s+", str(text or ""), maxsplit=1, flags=re.M)[0]
        match = re.search(r"\{\{(.*?)\}\}", raw, flags=re.S)
        if match:
            idea = " ".join(match.group(1).strip().split()).strip("` .;:-")
            if idea:
                return idea
        return None

    @classmethod
    def _contract_flags(cls, text: str) -> dict[str, bool]:
        prefix = re.split(r"```|^\s*def\s+", str(text or ""), maxsplit=1, flags=re.M)[0]
        ideas = [match.strip() for match in re.findall(r"\{\{(.*?)\}\}", prefix, flags=re.S) if match.strip()]
        exact = re.fullmatch(r"\s*\{\{.+?\}\}\s*```python\s*\n.*?```\s*", str(text or ""), flags=re.S)
        return {"idea_contract_success": len(ideas) == 1, "format_contract_success": len(ideas) == 1 and str(text or "").count("```") == 2 and exact is not None}

    @staticmethod
    def _trim_to_parseable_python(snippet: str) -> str:
        lines = str(snippet or "").splitlines()
        if not lines:
            return ""
        start = 0
        for idx, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith("import ") or stripped.startswith("from ") or stripped.startswith("def "):
                start = idx
                break
        candidate = "\n".join(lines[start:]).strip()
        while candidate:
            try:
                ast.parse(candidate)
                return candidate
            except SyntaxError as exc:
                lineno = getattr(exc, "lineno", None)
                current_lines = candidate.splitlines()
                if not lineno or lineno <= 0 or lineno > len(current_lines):
                    current_lines = current_lines[:-1]
                else:
                    current_lines = current_lines[: max(0, lineno - 1)]
                candidate = "\n".join(current_lines).rstrip()
            except Exception:
                return ""
        return ""

    @staticmethod
    def _strategy_has_metadata_leak(strategy: str) -> bool:
        lower = str(strategy or "").lower()
        forbidden = (
            "current_frontier_score",
            "population_best_score",
            "frontier_score",
            "parent_score",
            "parent_scores",
            "prompt_id",
            "parent_id",
            "reward",
            "raw score",
            "score=",
            "r=",
        )
        return any(fragment in lower for fragment in forbidden)

    @classmethod
    def extract_python_from_response(cls, response: str) -> str:
        text = str(response or "")
        blocks = list(re.finditer(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.I | re.S))
        if len(blocks) == 1:
            return cls._trim_to_parseable_python(blocks[0].group(1).strip())
        match = re.search(r"(?m)^\s*def\s+", text)
        if match:
            return cls._trim_to_parseable_python(text[match.start():])
        raw_function = SampleTrimmer.trim_preface_of_function(text)
        return cls._trim_to_parseable_python(raw_function or text)

    @staticmethod
    def _failure(
        failure_label: str,
        *,
        strategy: str | None = None,
        python: str | None = None,
    ) -> dict:
        return {
            "failure_label": str(failure_label),
            "strategy": strategy,
            "python": python,
            "func": None,
            "program": None,
        }
