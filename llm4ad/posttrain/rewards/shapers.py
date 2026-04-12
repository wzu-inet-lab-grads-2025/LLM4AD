from __future__ import annotations

from .base import RewardShaperBase


class NormalizeReward(RewardShaperBase):
    def transform(self, reward, *, context=None):
        return reward


class MarginReward(RewardShaperBase):
    def __init__(self, margin: float = 0.0):
        self._margin = margin

    def transform(self, reward, *, context=None):
        if reward is None:
            return None
        return reward - self._margin


class TimeoutPenalty(RewardShaperBase):
    def __init__(self, penalty: float = 1.0):
        self._penalty = penalty

    def transform(self, reward, *, context=None):
        if reward is None:
            return -self._penalty
        return reward
