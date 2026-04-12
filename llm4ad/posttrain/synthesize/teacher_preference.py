from __future__ import annotations

from .base import SynthesizeStrategyBase


class TeacherPreferenceSynthesizer(SynthesizeStrategyBase):
    def __init__(self, teacher_callable=None):
        self._teacher_callable = teacher_callable

    def synthesize(self, records, *, context=None):
        if self._teacher_callable is None:
            return []
        return self._teacher_callable(records=records, context=context)
