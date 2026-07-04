from __future__ import annotations

import copy
from typing import Dict, List

from ...base import Function


class EoHPrompt:
    _MAX_PARENT_DESCRIPTION_CHARS = 160
    _CONCISE_CODE_RULE = (
        "Keep the code concise and efficient. Prefer a short local decision rule over a long multi-stage procedure, "
        "and avoid unnecessary auxiliary structures or repeated full-input recomputation."
    )

    @classmethod
    def create_instruct_prompt(cls, prompt: str) -> List[Dict]:
        return [{"role": "user", "content": prompt}]

    @classmethod
    def get_system_prompt(cls) -> str:
        return ""

    @classmethod
    def build_prompt_record(
        cls,
        *,
        prompt_id: str,
        prompt: str,
        operator_type: str,
        parent_best_score: float | None = None,
        parent_codes: List[str] | None = None,
        parent_ids: List[int] | None = None,
        population_best_score: float | None = None,
        group_size: int = 1,
        reward_contract: str = "bqr_v1",
        system_prompt: str | None = None,
    ) -> Dict:
        user_prompt = str(prompt or "").strip()
        prompt_id_text = str(prompt_id or "").strip()
        operator_text = str(operator_type or "").strip()
        if not user_prompt:
            raise ValueError("EoH-RL prompt record missing prompt")
        if not prompt_id_text:
            raise ValueError("EoH-RL prompt record missing prompt_id")
        if not operator_text:
            raise ValueError("EoH-RL prompt record missing operator_type")
        system_prompt = cls.get_system_prompt() if system_prompt is None else str(system_prompt or "")
        return {
            "prompt_id": prompt_id_text,
            "prompt": user_prompt,
            "user_prompt": user_prompt,
            "system_prompt": system_prompt,
            "messages": cls._build_messages(system_prompt, user_prompt),
            "operator_type": operator_text,
            "parent_best_score": parent_best_score,
            "parent_codes": list(parent_codes or []),
            "parent_ids": None if parent_ids is None else list(parent_ids),
            "population_best_score": population_best_score,
            "group_size": int(group_size),
            "reward_contract": str(reward_contract or "bqr_v1"),
        }

    @staticmethod
    def _build_messages(system_prompt: str, prompt: str) -> List[Dict]:
        messages: List[Dict] = []
        if str(system_prompt or "").strip():
            messages.append({"role": "system", "content": str(system_prompt)})
        messages.append({"role": "user", "content": str(prompt or "")})
        return messages

    @staticmethod
    def _empty_template_function(template_function: Function) -> Function:
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ""
        return temp_func

    @classmethod
    def _clean_parent_description(cls, text: str) -> str:
        cleaned = " ".join(str(text or "").replace("`", " ").split()).strip()
        if not cleaned:
            return "No reliable algorithm description is available for this parent."
        if len(cleaned) <= cls._MAX_PARENT_DESCRIPTION_CHARS:
            return cleaned
        return cleaned[: max(0, cls._MAX_PARENT_DESCRIPTION_CHARS - 3)].rstrip() + "..."

    @staticmethod
    def _prepare_parent_code(indi: Function) -> str:
        display_func = copy.deepcopy(indi)
        display_func.docstring = ""
        lines = []
        keep_blank = False
        for line in str(display_func).splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if not stripped:
                if keep_blank or not lines:
                    continue
                lines.append("")
                keep_blank = True
                continue
            keep_blank = False
            lines.append(line.rstrip())
        return "\n".join(lines).strip()

    @classmethod
    def _format_parent(cls, idx: int, indi: Function) -> str:
        return (
            f"Algorithm {idx} description:\n"
            f"{cls._clean_parent_description(getattr(indi, 'algorithm', ''))}\n"
            f"Code:\n{cls._prepare_parent_code(indi)}"
        )

    @classmethod
    def _format_parents(cls, indivs: List[Function]) -> str:
        return "\n\n".join(cls._format_parent(i, indi) for i, indi in enumerate(indivs, start=1))

    @classmethod
    def _output_requirements(cls, template_function: Function) -> str:
        temp_func = cls._empty_template_function(template_function)
        return (
            "1. First, describe your new algorithm and main steps in one sentence.\n"
            "2. Next, implement the following Python function:\n"
            f"{str(temp_func)}\n"
            "Do not give additional explanations. Avoid long explanatory comments and unnecessary blank lines."
        )

    @classmethod
    def with_recovery_context(cls, prompt: str) -> str:
        return str(prompt or "").strip()

    @classmethod
    def get_prompt_i1(cls, task_prompt: str, template_function: Function) -> str:
        return (
            f"{task_prompt.strip()}\n"
            "Please create a new deterministic algorithm following the function template below.\n"
            f"{cls._output_requirements(template_function)}"
        )

    @classmethod
    def get_prompt_e1(cls, task_prompt: str, indivs: List[Function], template_function: Function) -> str:
        for indi in indivs:
            assert hasattr(indi, "algorithm")
        return (
            f"{task_prompt.strip()}\n"
            f"I have {len(indivs)} existing algorithms with their descriptions and codes as follows:\n"
            f"{cls._format_parents(indivs)}\n"
            "Please generate a new deterministic algorithm that is motivated by the following algorithms and may perform better on the same input.\n"
            "Prefer combining useful components from different parent algorithms instead of making only a very small edit of one parent.\n"
            f"{cls._CONCISE_CODE_RULE}\n"
            f"{cls._output_requirements(template_function)}"
        )

    @classmethod
    def get_prompt_e2(cls, task_prompt: str, indivs: List[Function], template_function: Function) -> str:
        for indi in indivs:
            assert hasattr(indi, "algorithm")
        return (
            f"{task_prompt.strip()}\n"
            f"I have {len(indivs)} existing algorithms with their descriptions and codes as follows:\n"
            f"{cls._format_parents(indivs)}\n"
            "Please generate a new deterministic algorithm that is motivated by the following algorithms and may perform better on the same input.\n"
            "You may first identify the strongest common backbone among them, and then revise one important fragment so the child is not only a tiny edit of a single parent.\n"
            f"{cls._CONCISE_CODE_RULE}\n"
            f"{cls._output_requirements(template_function)}"
        )

    @classmethod
    def get_prompt_m1(cls, task_prompt: str, indi: Function, template_function: Function) -> str:
        assert hasattr(indi, "algorithm")
        return (
            f"{task_prompt.strip()}\n"
            "I have one algorithm with its code as follows.\n"
            f"{cls._format_parent(1, indi)}\n"
            "Please modify this algorithm by introducing one new meaningful deterministic component into its main ranking, priority, or selection rule.\n"
            "Try to make the resulting decision behavior meaningfully different on the same input, rather than only making a very small cosmetic change.\n"
            f"{cls._CONCISE_CODE_RULE}\n"
            f"{cls._output_requirements(template_function)}"
        )

    @classmethod
    def get_prompt_m2(cls, task_prompt: str, indi: Function, template_function: Function) -> str:
        assert hasattr(indi, "algorithm")
        return (
            f"{task_prompt.strip()}\n"
            "I have one algorithm with its code as follows.\n"
            f"{cls._format_parent(1, indi)}\n"
            "Please identify one weak, rigid, or poorly differentiated fragment in the algorithm and replace it with a different deterministic rule derived from the current inputs.\n"
            "Try to change the main ranking, priority, or selection behavior instead of only adjusting a parameter or a superficial tie-breaker.\n"
            f"{cls._CONCISE_CODE_RULE}\n"
            f"{cls._output_requirements(template_function)}"
        )
