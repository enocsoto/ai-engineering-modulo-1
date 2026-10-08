"""Tests del cliente asíncrono. Los SDK están mockeados; no hay red."""

import asyncio
import os
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

import clients
from async_llm_manager import AsyncLLMManager, settings_from_env
from clients import (
    AnthropicClient,
    OpenAIAuthenticationError,
    OpenAIClient,
    OpenAIRateLimitError,
)
from schemas import ChatMessage, LLMSettings, ModelConfig, Provider

PRODUCTION_FILES = (
    "schemas.py",
    "clients.py",
    "async_llm_manager.py",
    "main.py",
)


def _user_messages(content: str = "¿Qué es la entropía?") -> list[ChatMessage]:
    return [ChatMessage(role="user", content=content)]


def _completion(text: str) -> MagicMock:
    return MagicMock(choices=[MagicMock(message=MagicMock(content=text))])


def _delta_chunk(content: object) -> MagicMock:
    chunk = MagicMock()
    chunk.choices = [MagicMock(delta=MagicMock(content=content))]
    return chunk


def _rate_limit_error() -> OpenAIRateLimitError:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(429, request=request)
    return OpenAIRateLimitError("rate limit", response=response, body=None)


def _auth_error() -> OpenAIAuthenticationError:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(401, request=request)
    return OpenAIAuthenticationError("invalid api key", response=response, body=None)


def _openai_manager(client: MagicMock) -> AsyncLLMManager:
    settings = LLMSettings(provider=Provider.OPENAI, openai_api_key="sk-test-value")
    with patch.object(clients, "AsyncOpenAI", return_value=client):
        return AsyncLLMManager(settings)


def _anthropic_manager(client: MagicMock) -> AsyncLLMManager:
    settings = LLMSettings(provider=Provider.ANTHROPIC, anthropic_api_key="sk-ant-test")
    with patch.object(clients, "AsyncAnthropic", return_value=client):
        return AsyncLLMManager(settings)


async def _collect(stream) -> list[str]:
    return [token async for token in stream]


class TestSchemas(unittest.TestCase):
    def test_source_has_no_time_sleep(self) -> None:
        for name in PRODUCTION_FILES:
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertNotIn("time.sleep", text, name)

    def test_rejects_invalid_message_config_and_blank_key(self) -> None:
        with self.assertRaises(ValidationError):
            ChatMessage(role="user", content="")
        with self.assertRaises(ValidationError):
            ModelConfig(temperature=2.5)
        with self.assertRaises(ValidationError):
            ModelConfig(max_tokens=0)
        with self.assertRaises(ValidationError):
            LLMSettings(provider=Provider.OPENAI, openai_api_key="   ")
        with self.assertRaises(ValidationError):
            LLMSettings(provider=Provider.ANTHROPIC, anthropic_api_key="")

    def test_blank_unused_key_becomes_none_and_secret_is_masked(self) -> None:
        settings = LLMSettings(
            provider=Provider.OPENAI,
            openai_api_key="sk-test-value",
            anthropic_api_key="  ",
        )
        self.assertIsNone(settings.anthropic_api_key)
        self.assertIsInstance(settings.openai_api_key, SecretStr)
        self.assertEqual(settings.active_api_key.get_secret_value(), "sk-test-value")
        self.assertEqual(settings.active_model, "gpt-4o-mini")
        self.assertNotIn("sk-test-value", str(settings))
        self.assertNotIn("sk-test-value", str(settings.openai_api_key))
        self.assertNotIn("sk-test-value", str(settings.model_dump()))
        self.assertNotIn("sk-test-value", str(settings.model_dump(mode="json")))
        self.assertNotIn("sk-test-value", settings.model_dump_json())


class TestClients(unittest.TestCase):
    def test_openai_generate_returns_message_content(self) -> None:
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=_completion("texto del modelo"))
        manager = _openai_manager(client)

        response = asyncio.run(manager.generate(_user_messages()))

        self.assertIsInstance(manager._client, OpenAIClient)
        self.assertIsNone(response.error)
        self.assertEqual(response.content, "texto del modelo")
        self.assertEqual(response.provider, "openai")
        kwargs = client.chat.completions.create.await_args.kwargs
        self.assertFalse(kwargs.get("stream", False))
        self.assertEqual(kwargs["messages"][0]["content"], "¿Qué es la entropía?")

    def test_openai_stream_yields_deltas_and_skips_empty(self) -> None:
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=_async_iter(
                _delta_chunk("hola"),
                _delta_chunk(None),
                _delta_chunk(""),
            )
        )
        manager = _openai_manager(client)

        tokens = asyncio.run(_collect(manager.stream(_user_messages("Di hola"))))

        self.assertEqual(tokens, ["hola"])
        self.assertTrue(client.chat.completions.create.await_args.kwargs["stream"])

    def test_anthropic_stream_yields_text_chunks(self) -> None:
        stream = MagicMock()
        stream.text_stream = _async_iter("ho", "la")
        context = AsyncMock()
        context.__aenter__.return_value = stream
        client = MagicMock()
        client.messages.stream.return_value = context
        manager = _anthropic_manager(client)

        tokens = asyncio.run(_collect(manager.stream(_user_messages("Di hola"))))

        self.assertEqual(tokens, ["ho", "la"])
        client.messages.stream.assert_called()

    def test_rate_limit_retries_then_returns_text(self) -> None:
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            side_effect=[_rate_limit_error(), _completion("ok")]
        )
        manager = _openai_manager(client)

        with patch.object(clients.asyncio, "sleep", new=AsyncMock()) as sleep:
            response = asyncio.run(manager.generate(_user_messages("Di hola")))

        self.assertIsNone(response.error)
        self.assertEqual(response.content, "ok")
        self.assertEqual(client.chat.completions.create.await_count, 2)
        self.assertEqual(sleep.await_count, 1)
        sleep.assert_awaited_with(0.5)

    def test_exhausted_retries_return_error_response(self) -> None:
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=_rate_limit_error())
        manager = _openai_manager(client)

        with patch.object(clients.asyncio, "sleep", new=AsyncMock()):
            response = asyncio.run(manager.generate(_user_messages("Di hola")))

        self.assertEqual(response.content, "")
        self.assertTrue(response.error)
        self.assertEqual(client.chat.completions.create.await_count, 3)

    def test_authentication_error_does_not_retry(self) -> None:
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=_auth_error())
        manager = _openai_manager(client)

        with patch.object(clients.asyncio, "sleep", new=AsyncMock()) as sleep:
            response = asyncio.run(manager.generate(_user_messages()))

        self.assertEqual(response.content, "")
        self.assertTrue(response.error)
        self.assertEqual(client.chat.completions.create.await_count, 1)
        self.assertEqual(sleep.await_count, 0)

    def test_manager_with_anthropic_uses_anthropic_client(self) -> None:
        block = MagicMock()
        block.text = "respuesta anthropic"
        client = MagicMock()
        client.messages.create = AsyncMock(return_value=MagicMock(content=[block]))
        manager = _anthropic_manager(client)

        response = asyncio.run(manager.generate(_user_messages("Di hola")))

        self.assertIsInstance(manager._client, AnthropicClient)
        client.messages.create.assert_awaited()
        self.assertIsNone(response.error)
        self.assertEqual(response.content, "respuesta anthropic")
        self.assertEqual(response.provider, "anthropic")

    def test_stream_does_not_retry_after_first_token(self) -> None:
        chunk = _delta_chunk("hola")
        calls = 0

        async def _create(*_args, **_kwargs):
            nonlocal calls
            calls += 1

            async def _chunks():
                yield chunk
                raise _rate_limit_error()

            return _chunks()

        client = MagicMock()
        client.chat.completions.create = _create
        manager = _openai_manager(client)

        with patch.object(clients.asyncio, "sleep", new=AsyncMock()) as sleep:
            tokens = asyncio.run(_collect(manager.stream(_user_messages("Di hola"))))

        self.assertEqual(tokens, ["hola"])
        self.assertEqual(calls, 1)
        self.assertEqual(sleep.await_count, 0)

    def test_unexpected_generate_error_is_a_response(self) -> None:
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))
        manager = _openai_manager(client)

        response = asyncio.run(manager.generate(_user_messages()))

        self.assertEqual(response.content, "")
        self.assertIn("inesperado", response.error or "")
        self.assertEqual(client.chat.completions.create.await_count, 1)

    def test_anthropic_splits_system_and_falls_back_to_user(self) -> None:
        client = MagicMock()
        client.messages.create = AsyncMock(return_value=MagicMock(content=[MagicMock(text="ok")]))
        manager = _anthropic_manager(client)
        mixed = [
            ChatMessage(role="system", content="uno"),
            ChatMessage(role="system", content="dos"),
            ChatMessage(role="user", content="pregunta"),
            ChatMessage(role="assistant", content="sigo"),
        ]
        asyncio.run(manager.generate(mixed))
        mixed_kwargs = client.messages.create.await_args.kwargs
        self.assertEqual(mixed_kwargs["system"], "uno\ndos")
        self.assertEqual(
            mixed_kwargs["messages"],
            [
                {"role": "user", "content": "pregunta"},
                {"role": "assistant", "content": "sigo"},
            ],
        )

        asyncio.run(manager.generate([ChatMessage(role="system", content="solo sistema")]))
        only_system = client.messages.create.await_args.kwargs
        self.assertNotIn("system", only_system)
        self.assertEqual(only_system["messages"], [{"role": "user", "content": "solo sistema"}])

    def test_openai_keeps_system_in_the_message_list(self) -> None:
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=_completion("ok"))
        manager = _openai_manager(client)
        asyncio.run(
            manager.generate(
                [
                    ChatMessage(role="system", content="contexto"),
                    ChatMessage(role="user", content="pregunta"),
                ]
            )
        )
        messages = client.chat.completions.create.await_args.kwargs["messages"]
        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "contexto"},
                {"role": "user", "content": "pregunta"},
            ],
        )
        self.assertEqual(client.chat.completions.create.await_args.kwargs["model"], "gpt-4o-mini")

    def test_settings_from_env_uses_provider_and_keys(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "anthropic",
                "OPENAI_API_KEY": "   ",
                "ANTHROPIC_API_KEY": "sk-ant-test",
            },
            clear=False,
        ):
            settings = settings_from_env()
        self.assertEqual(settings.provider, Provider.ANTHROPIC)
        self.assertIsNone(settings.openai_api_key)
        self.assertEqual(settings.active_api_key.get_secret_value(), "sk-ant-test")
        self.assertEqual(settings.active_model, "claude-3-5-haiku-latest")


async def _async_iter(*parts):
    for part in parts:
        yield part


if __name__ == "__main__":
    unittest.main()
