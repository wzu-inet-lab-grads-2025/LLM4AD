from __future__ import annotations


class CompositeReward:
    def __init__(self, provider, shapers=None):
        self._provider = provider
        self._shapers = shapers or []

    def score(self, sample, *, context=None):
        reward = self._provider.score(sample, context=context)
        for shaper in self._shapers:
            reward = shaper.transform(reward, context=context)
        return reward
