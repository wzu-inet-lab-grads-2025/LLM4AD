from __future__ import annotations


class ServeManagerBase:
    def start(self, version_id: str | None = None):
        raise NotImplementedError

    def switch(self, version_id: str):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError
