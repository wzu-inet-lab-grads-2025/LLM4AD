from __future__ import annotations

from llm4ad.posttrain.schemas import ParseEvent, SampleRecord


class MethodAdapterBase:
    def __init__(self, *, round_config=None, event_store=None):
        self._round_config = round_config
        self._event_store = event_store
        self._runtime = None
        self._method = None

    def bind(self, method, runtime, event_store=None) -> None:
        self._method = method
        self._runtime = runtime
        if event_store is not None:
            self._event_store = event_store

    def before_sample(self, **kwargs) -> None:
        return None

    def after_sample(self, **kwargs) -> None:
        return None

    def on_parse_failure(
        self,
        *,
        error_type: str,
        error_message: str | None = None,
        program_ok: bool = False,
        **kwargs,
    ) -> None:
        if self._runtime is None or self._event_store is None:
            return
        sample_id = self._ensure_sample_id()
        ctx = self._runtime.snapshot()
        self._event_store.append(
            ParseEvent(
                run_id=ctx.run_id,
                round_id=ctx.round_id,
                sample_id=sample_id,
                parse_ok=False,
                program_ok=program_ok,
                error_type=error_type,
                error_message=error_message,
            )
        )

    def on_program_failure(
        self, *, error_type: str, error_message: str | None = None, **kwargs
    ) -> None:
        if self._runtime is None or self._event_store is None:
            return
        sample_id = self._ensure_sample_id()
        ctx = self._runtime.snapshot()
        self._event_store.append(
            ParseEvent(
                run_id=ctx.run_id,
                round_id=ctx.round_id,
                sample_id=sample_id,
                parse_ok=True,
                program_ok=False,
                error_type=error_type,
                error_message=error_message,
            )
        )

    def after_evaluate(
        self,
        *,
        prompt=None,
        messages=None,
        response=None,
        function=None,
        program=None,
        algorithm=None,
        score=None,
        sample_time=None,
        eval_time=None,
        observation=None,
        image64=None,
        provenance=None,
        **kwargs,
    ) -> None:
        if self._runtime is None or self._event_store is None:
            return
        sample_id = self._ensure_sample_id()
        ctx = self._runtime.snapshot()
        self._event_store.append(
            SampleRecord(
                run_id=ctx.run_id,
                round_id=ctx.round_id,
                sample_id=sample_id,
                method_name=ctx.method_name,
                phase=ctx.phase,
                operator=ctx.operator,
                parents=ctx.parents,
                generation=ctx.generation,
                prompt=None if prompt is None else str(prompt),
                messages=messages,
                response=response,
                function=None if function is None else str(function),
                program=program,
                algorithm=algorithm,
                score=score,
                sample_time=sample_time,
                eval_time=eval_time,
                observation=observation,
                image64=image64,
                provenance=provenance,
            )
        )

    def after_register(self, **kwargs) -> None:
        return None

    def should_stop_round(self, method) -> bool:
        return False

    def snapshot_state(self, method) -> dict:
        return {}

    def restore_state(self, method, snapshot: dict) -> None:
        return None

    def _ensure_sample_id(self) -> str:
        ctx = self._runtime.snapshot()
        if ctx.sample_id:
            return ctx.sample_id
        return self._runtime.begin_sample()
