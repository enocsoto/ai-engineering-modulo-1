"""Tests for AsyncLLMManager. SDKs are mocked; nothing hits the network."""

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

try:
    import httpx2 as httpx
except ImportError:
    import httpx
from pydantic import SecretStr, ValidationError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import async_llm_manager as mgr


def _rate_limit_error():
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(429, request=request)
    return mgr.OpenAIRateLimitError("rate limit", response=response, body=None)


async def _tokens(*parts: str):
    for part in parts:
        yield part


class TestAsyncLLMManager(unittest.TestCase):
    def test_source_has_no_time_sleep(self) -> None:
        text = (ROOT / "async_llm_manager.py").read_text(encoding="utf-8")
        self.assertNotIn("time.sleep", text)

    def test_rejects_temperature_and_empty_prompt(self) -> None:
        with self.assertRaises(ValidationError):
            mgr.GenerationParams(prompt="hola", temperature=2.5)
        with self.assertRaises(ValidationError):
            mgr.GenerationParams(prompt="", temperature=0.2)
        with self.assertRaises(ValidationError):
            mgr.GenerationParams(prompt="hola", max_tokens=0)

    def test_secret_is_masked_and_blank_key_is_missing(self) -> None:
        settings = mgr.LLMSettings(provider=mgr.Provider.OPENAI, openai_api_key="sk-test-value")
        self.assertNotIn("sk-test-value", str(settings.openai_api_key))
        self.assertNotIn("sk-test-value", str(settings.model_dump(mode="json")))
        self.assertIsInstance(settings.openai_api_key, SecretStr)
        with self.assertRaises(ValidationError):
            mgr.LLMSettings(provider=mgr.Provider.ANTHROPIC, anthropic_api_key="  ")

    def test_openai_stream_yields_text_deltas(self) -> None:
        chunk = MagicMock()
        chunk.choices = [MagicMock(delta=MagicMock(content="hola"))]
        empty = MagicMock()
        empty.choices = [MagicMock(delta=MagicMock(content=None))]
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=_tokens(chunk, empty))

        settings = mgr.LLMSettings(provider=mgr.Provider.OPENAI, openai_api_key="sk-test-value")
        with patch.object(mgr, "AsyncOpenAI", return_value=client):
            manager = mgr.AsyncLLMManager(settings)
        params = mgr.GenerationParams(prompt="Di hola")
        tokens = asyncio.run(_collect(manager.stream(params)))

        self.assertEqual(tokens, ["hola"])
        client.chat.completions.create.assert_awaited()

    def test_anthropic_stream_yields_text(self) -> None:
        stream = MagicMock()
        stream.text_stream = _tokens("ho", "la")
        context = AsyncMock()
        context.__aenter__.return_value = stream
        client = MagicMock()
        client.messages.stream.return_value = context

        settings = mgr.LLMSettings(
            provider=mgr.Provider.ANTHROPIC,
            anthropic_api_key="sk-ant-test",
        )
        with patch.object(mgr, "AsyncAnthropic", return_value=client):
            manager = mgr.AsyncLLMManager(settings)
        tokens = asyncio.run(_collect(manager.stream(mgr.GenerationParams(prompt="Di hola"))))

        self.assertEqual(tokens, ["ho", "la"])

    def test_rate_limit_retries_then_streams(self) -> None:
        chunk = MagicMock()
        chunk.choices = [MagicMock(delta=MagicMock(content="ok"))]
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            side_effect=[_rate_limit_error(), _tokens(chunk)]
        )
        settings = mgr.LLMSettings(provider=mgr.Provider.OPENAI, openai_api_key="sk-test-value")
        with patch.object(mgr, "AsyncOpenAI", return_value=client):
            manager = mgr.AsyncLLMManager(settings)

        with patch.object(mgr.asyncio, "sleep", new=AsyncMock()):
            tokens = asyncio.run(_collect(manager.stream(mgr.GenerationParams(prompt="Di hola"))))

        self.assertEqual(tokens, ["ok"])
        self.assertEqual(client.chat.completions.create.await_count, 2)

    def test_exhausted_retries_do_not_raise(self) -> None:
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=_rate_limit_error())
        settings = mgr.LLMSettings(provider=mgr.Provider.OPENAI, openai_api_key="sk-test-value")
        with patch.object(mgr, "AsyncOpenAI", return_value=client):
            manager = mgr.AsyncLLMManager(settings)

        with patch.object(mgr.asyncio, "sleep", new=AsyncMock()):
            tokens = asyncio.run(_collect(manager.stream(mgr.GenerationParams(prompt="Di hola"))))

        self.assertEqual(tokens, [])
        self.assertEqual(client.chat.completions.create.await_count, mgr.MAX_ATTEMPTS)


async def _collect(stream):
    return [token async for token in stream]


if __name__ == "__main__":
    unittest.main()
