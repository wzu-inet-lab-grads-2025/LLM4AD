from __future__ import annotations

from typing import Any

from ...base import LLM


class LocalTransformersLLM(LLM):
    def __init__(
        self,
        model_path: str,
        *,
        tokenizer_path: str | None = None,
        device_map: str | dict | None = "auto",
        torch_dtype: str | None = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        use_chat_template: bool = True,
        trust_remote_code: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "LocalTransformersLLM requires torch and transformers. Use the project virtual environment and install requirements-posttrain.txt."
            ) from exc

        self._torch = torch
        self._model_path = model_path
        self._tokenizer_path = tokenizer_path or model_path
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._use_chat_template = use_chat_template
        self._device = self._resolve_device(device_map)

        model_kwargs = {
            "trust_remote_code": trust_remote_code,
        }
        if device_map is not None and self._can_use_device_map(device_map):
            model_kwargs["device_map"] = device_map
        resolved_dtype = self._resolve_torch_dtype(torch_dtype)
        if resolved_dtype is not None:
            model_kwargs["dtype"] = resolved_dtype

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._tokenizer_path,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            **model_kwargs,
        )
        if "device_map" not in model_kwargs and self._device is not None:
            self.model.to(self._device)
        self.model.eval()

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        messages = kwargs.get("messages")
        max_new_tokens = kwargs.get("max_new_tokens", self._max_new_tokens)
        temperature = kwargs.get("temperature", self._temperature)
        top_p = kwargs.get("top_p", self._top_p)

        model_inputs = self._build_inputs(prompt, messages=messages)
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if temperature is not None and temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p
        else:
            generation_kwargs["do_sample"] = False

        with self._torch.inference_mode():
            output_ids = self.model.generate(**model_inputs, **generation_kwargs)

        prompt_len = model_inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][prompt_len:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def close(self):
        del self.model
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def _build_inputs(self, prompt: str | Any, *, messages=None):
        text = self._build_text(prompt, messages=messages)
        encoded = self.tokenizer(text, return_tensors="pt")

        target_device = self._device
        if (
            target_device is None
            and hasattr(self.model, "hf_device_map")
            and self.model.hf_device_map
        ):
            first_device = next(iter(self.model.hf_device_map.values()))
            if isinstance(first_device, str) and first_device not in {"cpu", "disk"}:
                target_device = self._torch.device(first_device)
        elif target_device is None and hasattr(self.model, "device"):
            target_device = self.model.device

        if target_device is not None:
            encoded = {key: value.to(target_device) for key, value in encoded.items()}
        return encoded

    def _build_text(self, prompt: str | Any, *, messages=None) -> str:
        if messages is None and not isinstance(prompt, str):
            messages = prompt

        if messages is not None:
            if isinstance(messages, dict):
                messages = [messages]
            if self._use_chat_template and hasattr(
                self.tokenizer, "apply_chat_template"
            ):
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            return "\n".join(str(message) for message in messages)

        prompt_text = "" if prompt is None else str(prompt).strip()
        if self._use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt_text

    def _can_use_device_map(self, device_map) -> bool:
        if device_map is None or device_map == "cpu":
            return False
        try:
            import accelerate  # noqa: F401
        except ImportError:
            return False
        return True

    def _resolve_device(self, device_map):
        if isinstance(device_map, str) and device_map in {"cpu", "cuda"}:
            return self._torch.device(device_map)
        if self._torch.cuda.is_available():
            return self._torch.device("cuda")
        return self._torch.device("cpu")

    def _resolve_torch_dtype(self, torch_dtype):
        if torch_dtype is None:
            return None
        if torch_dtype == "auto":
            if self._device is not None and self._device.type == "cuda":
                return self._torch.float16
            return self._torch.float32
        if isinstance(torch_dtype, str) and hasattr(self._torch, torch_dtype):
            return getattr(self._torch, torch_dtype)
        return torch_dtype
