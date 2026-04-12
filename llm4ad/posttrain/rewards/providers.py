from __future__ import annotations

from .base import RewardProviderBase


class LocalEvaluatorProvider(RewardProviderBase):
    def score(self, sample, *, context=None):
        return (
            sample.get("score")
            if isinstance(sample, dict)
            else getattr(sample, "score", None)
        )


class ExternalVerifierProvider(RewardProviderBase):
    def __init__(self, verifier_callable=None):
        self._verifier_callable = verifier_callable

    def score(self, sample, *, context=None):
        if self._verifier_callable is None:
            raise NotImplementedError("External verifier provider requires a callable.")
        return self._verifier_callable(sample=sample, context=context)
