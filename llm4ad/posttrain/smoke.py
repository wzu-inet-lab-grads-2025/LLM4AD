from __future__ import annotations

import json
from pathlib import Path


class BestScoreExtractor:
    def __call__(
        self,
        *,
        search_report,
        workflow=None,
        model_ref=None,
        line_name=None,
        context=None,
    ):
        log_dir = None if search_report is None else search_report.get("log_dir")
        if not log_dir:
            event_score = self._extract_from_event_store(workflow)
            if event_score is not None:
                return {"score": event_score, "source": "event_store"}
            return {"score": None, "reason": "log_dir unavailable"}

        best_path = Path(log_dir) / "samples" / "samples_best.json"
        if best_path.exists():
            try:
                payload = json.loads(best_path.read_text(encoding="utf-8"))
                if isinstance(payload, list) and payload:
                    score = payload[-1].get("score")
                    return {"score": score, "source": str(best_path)}
                if isinstance(payload, dict):
                    return {"score": payload.get("score"), "source": str(best_path)}
            except json.JSONDecodeError:
                pass

        samples_dir = Path(log_dir) / "samples"
        best_score = None
        if samples_dir.exists():
            for sample_file in sorted(samples_dir.glob("samples_*.json")):
                try:
                    payload = json.loads(sample_file.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                entries = payload if isinstance(payload, list) else [payload]
                for entry in entries:
                    score = entry.get("score")
                    if not isinstance(score, (int, float)):
                        continue
                    if best_score is None or score > best_score:
                        best_score = score
        return {"score": best_score, "source": str(samples_dir)}

    def _extract_from_event_store(self, workflow):
        if (
            workflow is None
            or not hasattr(workflow, "event_store")
            or not hasattr(workflow, "runtime")
        ):
            return None
        run_id = workflow.runtime.run_id
        round_id = workflow.runtime.round_id
        best_score = None
        for payload in workflow.event_store.iter_events(
            "EvalTraceRecord", run_id=run_id, round_id=round_id
        ):
            score = payload.get("score")
            if not isinstance(score, (int, float)):
                continue
            if best_score is None or score > best_score:
                best_score = score
        if best_score is not None:
            return best_score
        for payload in workflow.event_store.iter_events(
            "SampleRecord", run_id=run_id, round_id=round_id
        ):
            score = payload.get("score")
            if not isinstance(score, (int, float)):
                continue
            if best_score is None or score > best_score:
                best_score = score
        return best_score


class WorkflowSmokeRunner:
    def __init__(self, workflow_factory, *, score_extractor=None):
        self._workflow_factory = workflow_factory
        self._score_extractor = score_extractor or BestScoreExtractor()

    def __call__(self, *, model_ref, line_name, context=None):
        workflow = self._workflow_factory(
            model_ref=model_ref, line_name=line_name, context=context
        )
        if workflow is None:
            return {"score": None, "reason": "workflow_factory returned None"}
        search_report = workflow.run_search_round(
            resume_path=self._resolve_resume_path(context)
        )
        extracted = self._score_extractor(
            search_report=search_report,
            workflow=workflow,
            model_ref=model_ref,
            line_name=line_name,
            context=context,
        )
        result = {
            "line_name": line_name,
            "model_version": None if model_ref is None else model_ref.get("version_id"),
            "search_report": search_report,
        }
        if isinstance(extracted, dict):
            result.update(extracted)
        else:
            result["score"] = extracted
        return result

    def _resolve_resume_path(self, context):
        if not isinstance(context, dict):
            return None
        return context.get("smoke_resume_path")
