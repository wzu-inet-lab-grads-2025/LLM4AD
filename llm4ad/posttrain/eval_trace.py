from __future__ import annotations

from .schemas import EvalTraceRecord, ParseEvent


class EvalTraceRecorder:
    def __init__(self, posttrain_runtime, event_store):
        self._runtime = posttrain_runtime
        self._event_store = event_store

    def record_parse_failure(
        self, *, error_type: str, error_message: str | None = None
    ) -> None:
        ctx = self._runtime.snapshot()
        self._event_store.append(
            ParseEvent(
                run_id=ctx.run_id,
                round_id=ctx.round_id,
                sample_id=ctx.sample_id or "",
                parse_ok=False,
                program_ok=False,
                error_type=error_type,
                error_message=error_message,
            )
        )

    def record_program_failure(
        self, *, error_type: str, error_message: str | None = None
    ) -> None:
        ctx = self._runtime.snapshot()
        self._event_store.append(
            ParseEvent(
                run_id=ctx.run_id,
                round_id=ctx.round_id,
                sample_id=ctx.sample_id or "",
                parse_ok=True,
                program_ok=False,
                error_type=error_type,
                error_message=error_message,
            )
        )

    def record_exec_failure(
        self,
        *,
        eval_time: float | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        self.record_eval_result(
            score=None,
            eval_time=eval_time,
            compile_ok=False,
            runtime_ok=None,
            timeout=False,
            error_type=error_type,
            error_message=error_message,
        )

    def record_evaluator_failure(
        self,
        *,
        eval_time: float | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        self.record_eval_result(
            score=None,
            eval_time=eval_time,
            compile_ok=True,
            runtime_ok=False,
            timeout=False,
            error_type=error_type,
            error_message=error_message,
        )

    def record_timeout(
        self, *, eval_time: float | None = None, error_message: str | None = None
    ) -> None:
        self.record_eval_result(
            score=None,
            eval_time=eval_time,
            compile_ok=None,
            runtime_ok=None,
            timeout=True,
            error_type="TimeoutError",
            error_message=error_message,
        )

    def record_eval_result(
        self,
        *,
        score,
        eval_time: float | None,
        compile_ok: bool | None,
        runtime_ok: bool | None,
        timeout: bool,
        error_type: str | None,
        error_message: str | None,
        score_breakdown: dict | None = None,
    ) -> None:
        ctx = self._runtime.snapshot()
        self._event_store.append(
            EvalTraceRecord(
                run_id=ctx.run_id,
                round_id=ctx.round_id,
                sample_id=ctx.sample_id or "",
                score=score,
                eval_time=eval_time,
                compile_ok=compile_ok,
                runtime_ok=runtime_ok,
                timeout=timeout,
                error_type=error_type,
                error_message=error_message,
                score_breakdown=score_breakdown,
            )
        )
