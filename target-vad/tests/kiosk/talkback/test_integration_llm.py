"""Layer 2 — LLM integration test (requires running llama.cpp server)."""

import pytest

from modes.talkback.llm import LlmClient

LLAMA_URL = "http://127.0.0.1:8080/v1"


@pytest.mark.integration
class TestLlmIntegration:
    @pytest.mark.asyncio
    async def test_ping_server(self):
        client = LlmClient(base_url=LLAMA_URL, model="test")
        available = await client.ping()
        if not available:
            pytest.skip("llama.cpp server not running at " + LLAMA_URL)
        assert available
        await client.close()

    @pytest.mark.asyncio
    async def test_stream_tokens(self):
        client = LlmClient(base_url=LLAMA_URL, model="test")
        available = await client.ping()
        if not available:
            pytest.skip("llama.cpp server not running")

        messages = [
            {"role": "system", "content": "Reply in one word."},
            {"role": "user", "content": "Say hello."},
        ]
        tokens = []
        async for token in client.stream(messages):
            tokens.append(token)
            if len(tokens) > 20:
                break
        assert len(tokens) > 0
        await client.close()
