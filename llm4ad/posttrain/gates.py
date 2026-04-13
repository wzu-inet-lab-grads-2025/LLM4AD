from __future__ import annotations


class StaticValidationGate:
    def evaluate(
        self, candidate, active, base, *, validation_spec=None, context=None
    ) -> dict:
        if callable(validation_spec):
            result = validation_spec(
                candidate=candidate, active=active, base=base, context=context
            )
            return {"passed": bool(result.get("passed", False)), "details": result}
        return {
            "passed": False,
            "details": {
                "reason": "No static validation callable was provided.",
            },
        }


class SmokeSearchEvaluator:
    def __init__(
        self,
        runner_callable,
        *,
        compare_lines=("candidate", "active", "base"),
        min_candidate_margin: float = 0.0,
        maximize_metric: bool = True,
    ):
        self._runner_callable = runner_callable
        self._compare_lines = compare_lines
        self._min_candidate_margin = min_candidate_margin
        self._maximize_metric = maximize_metric

    def __call__(self, *, candidate, active, base, context=None):
        refs = {
            "candidate": candidate,
            "active": active,
            "base": base,
        }
        details = {}
        scores = {}

        for line_name in self._compare_lines:
            ref = refs.get(line_name)
            if ref is None:
                details[line_name] = {
                    "score": None,
                    "result": None,
                    "reason": "line unavailable",
                }
                continue
            result = self._runner_callable(
                model_ref=ref, line_name=line_name, context=context
            )
            score = self._extract_score(result)
            scores[line_name] = score
            details[line_name] = {"score": score, "result": result}

        candidate_score = scores.get("candidate")
        if candidate_score is None:
            return {
                "passed": False,
                "details": details,
                "reason": "candidate score unavailable",
            }

        competitors = [name for name in self._compare_lines if name != "candidate"]
        comparisons = {}
        passed = True
        for line_name in competitors:
            other_score = scores.get(line_name)
            if other_score is None:
                comparisons[line_name] = None
                continue
            if self._maximize_metric:
                ok = candidate_score >= other_score + self._min_candidate_margin
            else:
                ok = candidate_score <= other_score - self._min_candidate_margin
            comparisons[line_name] = ok
            passed = passed and ok

        return {
            "passed": passed,
            "scores": scores,
            "comparisons": comparisons,
            "details": details,
        }

    def _extract_score(self, result):
        if isinstance(result, (int, float)):
            return float(result)
        if isinstance(result, dict):
            for key in ("score", "best_score", "metric"):
                value = result.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
        return None


class SmokeSearchGate:
    def evaluate(
        self, candidate, active, base, *, smoke_spec=None, context=None
    ) -> dict:
        if callable(smoke_spec):
            result = smoke_spec(
                candidate=candidate, active=active, base=base, context=context
            )
            return {"passed": bool(result.get("passed", False)), "details": result}
        return {
            "passed": False,
            "details": {
                "reason": "No smoke-search callable was provided.",
            },
        }


class PromotionGate:
    def __init__(self, static_gate=None, smoke_gate=None):
        self._static_gate = static_gate or StaticValidationGate()
        self._smoke_gate = smoke_gate or SmokeSearchGate()

    def evaluate(
        self,
        candidate,
        active,
        base,
        *,
        validation_spec=None,
        smoke_spec=None,
        context=None,
    ) -> dict:
        static_result = self._static_gate.evaluate(
            candidate,
            active,
            base,
            validation_spec=validation_spec,
            context=context,
        )
        smoke_result = self._smoke_gate.evaluate(
            candidate,
            active,
            base,
            smoke_spec=smoke_spec,
            context=context,
        )
        return {
            "passed": static_result["passed"] and smoke_result["passed"],
            "static": static_result,
            "smoke": smoke_result,
        }
