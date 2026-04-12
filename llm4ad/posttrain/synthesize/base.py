from __future__ import annotations


class SynthesizeStrategyBase:
    def synthesize(self, records, *, context=None):
        raise NotImplementedError
