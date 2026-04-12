from __future__ import annotations


class RewardProviderBase:
    def score(self, sample, *, context=None):
        raise NotImplementedError


class RewardShaperBase:
    def transform(self, reward, *, context=None):
        raise NotImplementedError
