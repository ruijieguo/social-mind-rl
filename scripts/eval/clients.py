"""OpenAI-compatible chat client for DashScope, DeepSeek, local vLLM, or generic OpenAI."""
from __future__ import annotations
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI


_BACKEND_BASE_URLS = {
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "deepseek": "https://api.deepseek.com",
}


@dataclass
class BackendSpec:
    name: str               # "dashscope" | "deepseek" | "openai"
    model: str              # model id
    base_url: Optional[str] = None  # required for "openai", else derived
    extra_body: dict = field(default_factory=dict)
    api_key_env: Optional[str] = None  # env var name for api key

    def resolve_base_url(self) -> str:
        if self.base_url:
            return self.base_url
        if self.name in _BACKEND_BASE_URLS:
            return _BACKEND_BASE_URLS[self.name]
        raise ValueError(f"backend {self.name!r} requires base_url")

    def resolve_api_key(self) -> str:
        if self.api_key_env:
            v = os.environ.get(self.api_key_env, "")
            if v:
                return v
        defaults = {
            "dashscope": "DASHSCOPE_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "openai": "OPENAI_API_KEY",
        }
        env = defaults.get(self.name, "OPENAI_API_KEY")
        return os.environ.get(env, "")


@dataclass
class ChatResult:
    content: str
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ChatClient:
    """Thin wrapper around OpenAI client with retry + extra_body support."""

    def __init__(
        self,
        spec: BackendSpec,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff_base: float = 1.5,
    ):
        self.spec = spec
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._client = OpenAI(
            api_key=api_key if api_key is not None else spec.resolve_api_key(),
            base_url=spec.resolve_base_url(),
        )

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 32,
        n: int = 1,
        extra_body_override: Optional[dict] = None,
    ) -> ChatResult:
        body = dict(self.spec.extra_body)
        if extra_body_override:
            body.update(extra_body_override)

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.spec.model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    n=n,
                    timeout=self.timeout,
                    extra_body=body if body else None,
                )
                content = resp.choices[0].message.content or ""
                usage = getattr(resp, "usage", None) or type("U", (), {})()
                return ChatResult(
                    content=content,
                    total_tokens=getattr(usage, "total_tokens", 0) or 0,
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                )
            except Exception as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.backoff_base ** attempt)
                    continue
        assert last_err is not None
        raise last_err
