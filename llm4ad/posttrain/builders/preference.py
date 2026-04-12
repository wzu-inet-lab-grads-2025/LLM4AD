from __future__ import annotations

from collections import defaultdict

from .base import DatasetBuilderBase


def _scalar_score(score):
    if isinstance(score, (int, float)):
        return float(score)
    return None


class PreferenceBuilder(DatasetBuilderBase):
    def build(self, records, *, config, context):
        grouped = defaultdict(list)
        include_init = getattr(config, "include_init_phase", False)

        for record in records:
            if record.phase == "init" and not include_init:
                continue
            scalar = _scalar_score(record.score)
            if scalar is None:
                continue
            if not record.prompt:
                continue
            completion = record.response or record.function
            if not completion:
                continue
            grouped[record.prompt].append((scalar, completion, record))

        examples = []
        for prompt, items in grouped.items():
            if len(items) < 2:
                continue
            items.sort(key=lambda item: item[0], reverse=True)
            chosen_score, chosen_completion, chosen_record = items[0]
            rejected_score, rejected_completion, rejected_record = items[-1]
            if chosen_score <= rejected_score:
                continue
            examples.append(
                {
                    "prompt": prompt,
                    "chosen": chosen_completion,
                    "rejected": rejected_completion,
                    "chosen_score": chosen_score,
                    "rejected_score": rejected_score,
                    "margin": chosen_score - rejected_score,
                    "chosen_sample_id": chosen_record.sample_id,
                    "rejected_sample_id": rejected_record.sample_id,
                }
            )
        return examples
