from __future__ import annotations

from llm4ad.tools.llm.local_vllm.vllm_local_http_api import VLLMManager

from .base import ServeManagerBase


class VLLMServeManager(ServeManagerBase):
    def __init__(
        self,
        model_router,
        *,
        tokenizer_path: str | None = None,
        gpus=None,
        ports=None,
        gpu_mem_utils=None,
    ):
        self._model_router = model_router
        self._tokenizer_path = tokenizer_path
        self._gpus = gpus or []
        self._ports = ports or []
        self._gpu_mem_utils = gpu_mem_utils
        self._active_version = None
        self._manager = None

    def start(self, version_id: str | None = None):
        if version_id is None:
            return None
        return self.switch(version_id)

    def switch(self, version_id: str):
        record = self._model_router.resolve_version(version_id)
        if record is None:
            raise ValueError(f"Unable to resolve model version: {version_id}")
        self.stop()
        if not self._gpus or not self._ports:
            self._active_version = version_id
            return record

        tokenizer_path = self._tokenizer_path or record["path"]
        self._manager = VLLMManager()
        self._manager.deploy_models(
            model_path=record["path"],
            tknz_path=tokenizer_path,
            gpus=self._gpus,
            ports=self._ports,
            gpu_mem_utils=self._gpu_mem_utils,
        )
        self._active_version = version_id
        return record

    def stop(self):
        if self._manager is not None:
            self._manager.release_resources()
            self._manager = None
