from __future__ import annotations

import time
from typing import Any

from llm4ad.base import LLM

from .schemas import LLMRequestEvent, LLMResponseEvent


class PostTrainLLMProxy(LLM):
    def __init__(
        self,
        inner_llm: LLM,
        posttrain_runtime,
        event_store,
        model_id: str | None = None,
    ):
        super().__init__(
            do_auto_trim=inner_llm.do_auto_trim, debug_mode=inner_llm.debug_mode
        )
        self._inner_llm = inner_llm
        self._runtime = posttrain_runtime
        self._event_store = event_store
        self._model_id = model_id or getattr(inner_llm, "model", None)

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        sample_id = self._runtime.begin_sample()
        request_event = self._build_request_event(sample_id, prompt, kwargs)
        self._event_store.append(request_event)

        start = time.time()
        raw_response = self._inner_llm.draw_sample(prompt, *args, **kwargs)
        latency_seconds = time.time() - start

        self._event_store.append(
            LLMResponseEvent(
                run_id=request_event.run_id,
                round_id=request_event.round_id,
                sample_id=sample_id,
                raw_response=raw_response,
                latency_seconds=latency_seconds,
            )
        )
        return raw_response

    def draw_samples(self, prompts: list[str | Any], *args, **kwargs) -> list[str]:
        return [self.draw_sample(prompt, *args, **kwargs) for prompt in prompts]

    def close(self) -> None:
        self._inner_llm.close()

    def _build_request_event(
        self, sample_id: str, prompt: str | Any, kwargs: dict[str, Any]
    ) -> LLMRequestEvent:
        ctx = self._runtime.snapshot()
        prompt_text = None if prompt is None else str(prompt)
        messages = kwargs.get("messages")
        image64s = kwargs.get("image64s")
        sampling_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in {"messages", "image64s"}
        }
        return LLMRequestEvent(
            run_id=ctx.run_id,
            round_id=ctx.round_id,
            sample_id=sample_id,
            phase=ctx.phase,
            prompt=prompt_text,
            messages=messages,
            image64s=image64s,
            model_id=self._model_id,
            sampling_kwargs=sampling_kwargs,
        )
