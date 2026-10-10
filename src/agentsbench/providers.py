"""Optional OpenAI-compatible HTTP adapter, using only Python's standard library."""

import json
import os
from urllib.request import Request, urlopen


class OpenAICompatibleModel:
    def __init__(self, api_key: str | None = None, model: str | None = None, base_url: str | None = None, request_options: dict | None = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.request_options = request_options or {}
        if not self.api_key:
            raise ValueError("set OPENAI_API_KEY before using the API provider")
        if not self.base_url.startswith("https://"):
            raise ValueError("OPENAI_BASE_URL must use HTTPS")

    def complete(self, system: str, user: str) -> str:
        payload = json.dumps({
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            **self.request_options,
        }).encode("utf-8")
        request = Request(
            self.base_url + "/chat/completions",
            data=payload,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=120) as response:
            result = json.load(response)
        return result["choices"][0]["message"]["content"]


class MiniMaxModel(OpenAICompatibleModel):
    """MiniMax's China-region OpenAI-compatible chat endpoint."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        key = api_key or os.environ.get("MINIMAX_API_KEY")
        if not key:
            raise ValueError("set MINIMAX_API_KEY before using the MiniMax provider")
        super().__init__(
            api_key=key,
            model=model or os.environ.get("MINIMAX_MODEL", "MiniMax-M3"),
            base_url="https://api.minimaxi.com/v1",
            request_options={"max_tokens": 1024, "thinking": {"type": "disabled"}},
        )

