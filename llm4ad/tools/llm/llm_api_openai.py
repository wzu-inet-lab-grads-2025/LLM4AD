# This file is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
# Last Revision: 2025/2/16
#
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
#
# Permission is granted to use the LLM4AD platform for research purposes.
# All publications, software, or other works that utilize this platform
# or any part of its codebase must acknowledge the use of "LLM4AD" and
# cite the following reference:
#
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang,
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
#
# For inquiries regarding commercial use or licensing, please contact
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------
from __future__ import annotations

import openai
from typing import Any

from llm4ad.base import LLM


class OpenAIAPI(LLM):
    def __init__(
            self,
            base_url: str,
            api_key: str,
            model: str,
            timeout=60,
            generation_kwargs: dict[str, Any] | None = None,
            client_kwargs: dict[str, Any] | None = None,
            **kwargs,
    ):
        do_auto_trim = kwargs.pop('do_auto_trim', True)
        debug_mode = kwargs.pop('debug_mode', False)
        super().__init__(do_auto_trim=do_auto_trim, debug_mode=debug_mode)
        request_kwargs = dict(generation_kwargs or {})
        for key in ('max_tokens', 'temperature', 'top_p', 'frequency_penalty', 'presence_penalty', 'seed'):
            if key in kwargs:
                request_kwargs[key] = kwargs.pop(key)
        openai_client_kwargs = dict(client_kwargs or {})
        openai_client_kwargs.update(kwargs)
        self._base_url = base_url
        self._model = model
        self._timeout = timeout
        self._request_kwargs = {key: value for key, value in request_kwargs.items() if value is not None}
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, **openai_client_kwargs)

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        if isinstance(prompt, str):
            prompt = [{'role': 'user', 'content': prompt.strip()}]
        request_kwargs = dict(self._request_kwargs)
        for key in ('max_tokens', 'temperature', 'top_p', 'frequency_penalty', 'presence_penalty', 'seed'):
            if key in kwargs:
                request_kwargs[key] = kwargs[key]
        response = self._client.chat.completions.create(
            model=self._model,
            messages=prompt,
            stream=False,
            **request_kwargs,
        )
        return response.choices[0].message.content
