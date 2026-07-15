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
        parent_best_profile: List[float] | None = None,
        parent_best_id: int | None = None,
        parent_codes: List[str] | None = None,
        parent_ids: List[int] | None = None,
        population_best_score: float | None = None,
        group_size: int = 1,
        reward_contract: str = "vc_pair_v1",
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
            "system_prompt": system_prompt,
            "messages": cls._build_messages(system_prompt, user_prompt),
            "operator_type": operator_text,
            "parent_best_score": parent_best_score,
            "parent_best_profile": None if parent_best_profile is None else list(parent_best_profile),
            "parent_best_id": parent_best_id,
            "parent_codes": list(parent_codes or []),
            "parent_ids": None if parent_ids is None else list(parent_ids),
            "population_best_score": population_best_score,
            "group_size": int(group_size),
            "reward_contract": str(reward_contract or "vc_pair_v1"),
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
            "Return exactly two parts:\n\n"
            "{{The idea of the algorithm is to <one concise sentence describing the deterministic decision rule>.}}\n\n"
            "```python\n"
            f"{str(temp_func).strip()}\n"
            "```\n\n"
            "Do not include any other text. The code block must contain exactly one complete implementation of the required function. "
            "The algorithm must be deterministic. Do not use randomness, time, external state, hidden evaluation information, or unavailable imports. "
            "Keep the implementation concise and focused on the main ranking, priority, or selection rule."
        )

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
            "Please generate a new deterministic algorithm motivated by the following parent algorithms.\n"
            "Use the strongest parent as the performance anchor, and borrow one structurally different decision idea from the other parent.\n"
            "The child should combine useful components from both parents and produce a meaningfully different ranking, priority, or selection behavior, rather than being a tiny edit of either parent.\n"
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
            "Please generate a new deterministic algorithm motivated by the following parent algorithms.\n"
            "Keep the strongest reliable backbone among the parents, then replace or redesign one important decision fragment using a structurally different idea from the other parent.\n"
            "The child should preserve the useful core behavior while changing one key ranking, priority, or selection mechanism.\n"
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
            "Please modify this algorithm by injecting one new deterministic component into its main ranking, priority, or selection rule.\n"
            "The added component should be derived from the current inputs and should introduce a meaningful new signal or interaction, not just a constant tweak or cosmetic edit.\n"
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
            "Please identify one weak, rigid, overly simple, or poorly differentiated fragment in the algorithm and replace it with a different deterministic rule.\n"
            "The replacement should be input-derived and should change the main decision behavior while keeping the implementation concise.\n"
            f"{cls._CONCISE_CODE_RULE}\n"
            f"{cls._output_requirements(template_function)}"
        )
