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
    def __init__(
        self,
        workflow_factory,
        *,
        score_extractor=None,
        compare_lines=("candidate", "active", "base"),
        min_candidate_margin: float = 0.0,
        maximize_metric: bool = True,
    ):
        self._workflow_factory = workflow_factory
        self._score_extractor = score_extractor or BestScoreExtractor()
        self._compare_lines = compare_lines
        self._min_candidate_margin = min_candidate_margin
        self._maximize_metric = maximize_metric

    def __call__(
        self,
        *,
        model_ref=None,
        line_name=None,
        candidate=None,
        active=None,
        base=None,
        context=None,
    ):
        if line_name is not None or model_ref is not None:
            return self._run_single_line(
                model_ref=model_ref,
                line_name=line_name,
                context=context,
            )

        refs = {
            "candidate": candidate,
            "active": active,
            "base": base,
        }
        details = {}
        scores = {}

        for current_line in self._compare_lines:
            ref = refs.get(current_line)
            if ref is None:
                details[current_line] = {
                    "score": None,
                    "result": None,
                    "reason": "line unavailable",
                }
                continue
            result = self._run_single_line(
                model_ref=ref,
                line_name=current_line,
                context=context,
            )
            details[current_line] = result
            score = result.get("score") if isinstance(result, dict) else None
            scores[current_line] = score

        candidate_score = scores.get("candidate")
        if candidate_score is None:
            return {
                "passed": False,
                "scores": scores,
                "details": details,
                "reason": "candidate score unavailable",
            }

        comparisons = {}
        passed = True
        for current_line in self._compare_lines:
            if current_line == "candidate":
                continue
            other_score = scores.get(current_line)
            if other_score is None:
                comparisons[current_line] = None
                continue
            if self._maximize_metric:
                ok = candidate_score >= other_score + self._min_candidate_margin
            else:
                ok = candidate_score <= other_score - self._min_candidate_margin
            comparisons[current_line] = ok
            passed = passed and ok

        return {
            "passed": passed,
            "scores": scores,
            "comparisons": comparisons,
            "details": details,
        }

    def _run_single_line(self, *, model_ref, line_name, context=None):
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
