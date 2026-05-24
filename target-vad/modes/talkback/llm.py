"""LLM client — OpenAI-compatible HTTP streaming against llama.cpp server.

Connects to a local llama.cpp server's /v1/chat/completions endpoint with
stream=true. Supports cancellation for barge-in.
"""

import json
from typing import AsyncIterator

import aiohttp


class LlmClient:
    """Streaming LLM client using the OpenAI chat completions API."""

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.6,
        max_tokens: int = 512,
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._session: aiohttp.ClientSession | None = None
        self._cancelled = False

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        self._cancelled = False
        session = await self._ensure_session()

        payload = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

        async with session.post(
            f"{self._base_url}/chat/completions",
            json=payload,
        ) as resp:
            async for line in resp.content:
                if self._cancelled:
                    return
                line_str = line.decode("utf-8").strip()
                if not line_str.startswith("data: "):
                    continue
                data = line_str[6:]
                if data == "[DONE]":
                    return
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    def cancel(self) -> None:
        self._cancelled = True

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def ping(self) -> bool:
        try:
            session = await self._ensure_session()
            async with session.get(f"{self._base_url}/models") as resp:
                return resp.status == 200
        except (aiohttp.ClientError, OSError):
            return False
