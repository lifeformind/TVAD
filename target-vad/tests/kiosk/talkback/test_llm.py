"""Tests for LLM client — OpenAI-compatible HTTP streaming."""

import json
from unittest.mock import MagicMock, patch

import pytest

from modes.talkback.llm import LlmClient


class FakeStreamResponse:
    """Simulates an aiohttp SSE response for streaming chat completions."""

    def __init__(self, tokens: list[str]):
        self._tokens = tokens
        self.status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    @property
    def content(self):
        return self

    async def __aiter__(self):
        for token in self._tokens:
            chunk = {
                "choices": [{"delta": {"content": token}, "finish_reason": None}]
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"


class TestLlmClientStream:
    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        client = LlmClient(
            base_url="http://fake:8080/v1",
            model="test-model",
            temperature=0.6,
            max_tokens=512,
        )
        messages = [{"role": "user", "content": "hello"}]

        fake_resp = FakeStreamResponse(["Hello", " world", "!"])
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(return_value=fake_resp)
        client._session = mock_session

        tokens = []
        async for token in client.stream(messages):
            tokens.append(token)

        assert tokens == ["Hello", " world", "!"]

    @pytest.mark.asyncio
    async def test_cancel_stops_iteration(self):
        client = LlmClient(
            base_url="http://fake:8080/v1",
            model="test-model",
        )
        messages = [{"role": "user", "content": "hi"}]

        fake_resp = FakeStreamResponse(["a", "b", "c", "d", "e"])
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(return_value=fake_resp)
        client._session = mock_session

        tokens = []
        async for token in client.stream(messages):
            tokens.append(token)
            if len(tokens) == 2:
                client.cancel()
                break
        assert len(tokens) == 2


class TestLlmClientInit:
    def test_default_values(self):
        client = LlmClient(base_url="http://localhost:8080/v1", model="qwen")
        assert client._model == "qwen"
        assert client._temperature == 0.6
        assert client._max_tokens == 512
