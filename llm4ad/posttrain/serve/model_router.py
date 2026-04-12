from __future__ import annotations


class RegistryModelRouter:
    def __init__(self, registry):
        self._registry = registry

    def resolve_version(self, version_id: str):
        for getter in (
            self._registry.get_candidate,
            self._registry.get_active,
            self._registry.get_previous,
        ):
            record = getter()
            if record is not None and record.get("version_id") == version_id:
                return record
        return None
