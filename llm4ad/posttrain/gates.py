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
