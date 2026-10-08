"""Clientes asíncronos de OpenAI y Anthropic, con reintentos."""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from anthropic import APIConnectionError as AnthropicAPIConnectionError
from anthropic import APITimeoutError as AnthropicAPITimeoutError
from anthropic import AsyncAnthropic
from anthropic import AuthenticationError as AnthropicAuthenticationError
from anthropic import RateLimitError as AnthropicRateLimitError
from openai import APIConnectionError as OpenAIAPIConnectionError
from openai import APITimeoutError as OpenAIAPITimeoutError
from openai import AsyncOpenAI
from openai import AuthenticationError as OpenAIAuthenticationError
from openai import RateLimitError as OpenAIRateLimitError

from schemas import ChatMessage, ModelConfig, ModelResponse

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
_BASE_DELAY_SECONDS = 0.5

RETRYABLE_ERRORS = (
    OpenAIAPIConnectionError,
    OpenAIAPITimeoutError,
    OpenAIRateLimitError,
    AnthropicAPIConnectionError,
    AnthropicAPITimeoutError,
    AnthropicRateLimitError,
)
AUTH_ERRORS = (OpenAIAuthenticationError, AnthropicAuthenticationError)

_AUTH_MESSAGE = "La clave de API es inválida."
_EXHAUSTED_MESSAGE = (
    "Se agotaron los reintentos por un fallo de red, tiempo de espera o límite de uso."
)


def _unexpected_message(exc: Exception) -> str:
    return f"Error inesperado al llamar al modelo ({type(exc).__name__})."


def _openai_messages(messages: list[ChatMessage]) -> list[dict[str, str]]:
    return [{"role": message.role, "content": message.content} for message in messages]


def _anthropic_payload(
    messages: list[ChatMessage],
) -> tuple[str | None, list[dict[str, str]]]:
    system_parts = [message.content for message in messages if message.role == "system"]
    dialogue = [
        {"role": message.role, "content": message.content}
        for message in messages
        if message.role != "system"
    ]
    if not dialogue and system_parts:
        return None, [{"role": "user", "content": "\n".join(system_parts)}]
    system = "\n".join(system_parts) if system_parts else None
    return system, dialogue


def _anthropic_kwargs(
    messages: list[ChatMessage],
    config: ModelConfig,
    model: str,
) -> dict[str, object]:
    system, dialogue = _anthropic_payload(messages)
    payload: dict[str, object] = {
        "model": model,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "messages": dialogue,
    }
    if system is not None:
        payload["system"] = system
    return payload


def _anthropic_text(response: object) -> str:
    blocks = getattr(response, "content", None) or []
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, str):
            parts.append(block)
            continue
        if isinstance(block, dict):
            text = block.get("text")
        else:
            text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


class BaseLLMClient(ABC):
    """Llamada normal o streaming. Los fallos de red no se propagan."""

    provider_name: str

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._default_model = model

    def _resolve(self, config: ModelConfig | None) -> tuple[ModelConfig, str]:
        resolved = config if config is not None else ModelConfig()
        return resolved, resolved.model or self._default_model

    def _failure(self, model: str, error: str) -> ModelResponse:
        return ModelResponse(
            content="",
            model=model,
            provider=self.provider_name,
            error=error,
        )

    async def generate(
        self,
        messages: list[ChatMessage],
        config: ModelConfig | None = None,
    ) -> ModelResponse:
        resolved, model_name = self._resolve(config)
        delay = _BASE_DELAY_SECONDS
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return await self._generate_once(messages, resolved, model_name)
            except AUTH_ERRORS as exc:
                logger.error("clave rechazada (%s); no se reintenta", type(exc).__name__)
                return self._failure(model_name, _AUTH_MESSAGE)
            except RETRYABLE_ERRORS as exc:
                logger.warning(
                    "fallo de red o rate limit (%s), intento %s de %s",
                    type(exc).__name__,
                    attempt,
                    MAX_ATTEMPTS,
                )
                if attempt == MAX_ATTEMPTS:
                    logger.error("se agotaron los reintentos; se continúa")
                    return self._failure(model_name, _EXHAUSTED_MESSAGE)
                await asyncio.sleep(delay)
                delay *= 2
            except Exception as exc:
                logger.error("error inesperado en generate (%s)", type(exc).__name__)
                return self._failure(model_name, _unexpected_message(exc))
        return self._failure(model_name, _EXHAUSTED_MESSAGE)

    async def stream(
        self,
        messages: list[ChatMessage],
        config: ModelConfig | None = None,
    ) -> AsyncIterator[str]:
        resolved, model_name = self._resolve(config)
        delay = _BASE_DELAY_SECONDS
        for attempt in range(1, MAX_ATTEMPTS + 1):
            started = False
            try:
                async for token in self._stream_once(messages, resolved, model_name):
                    started = True
                    yield token
                return
            except AUTH_ERRORS as exc:
                logger.error("clave rechazada (%s); no se reintenta", type(exc).__name__)
                return
            except RETRYABLE_ERRORS as exc:
                if started:
                    logger.error(
                        "fallo durante el streaming (%s); se corta esta respuesta y se continúa",
                        type(exc).__name__,
                    )
                    return
                logger.warning(
                    "fallo de red o rate limit (%s), intento %s de %s",
                    type(exc).__name__,
                    attempt,
                    MAX_ATTEMPTS,
                )
                if attempt == MAX_ATTEMPTS:
                    logger.error("se agotaron los reintentos; se continúa")
                    return
                await asyncio.sleep(delay)
                delay *= 2
            except Exception as exc:
                logger.error("error inesperado en stream (%s)", type(exc).__name__)
                return

    @abstractmethod
    async def _generate_once(
        self,
        messages: list[ChatMessage],
        config: ModelConfig,
        model: str,
    ) -> ModelResponse:
        """Una llamada sin streaming."""

    @abstractmethod
    def _stream_once(
        self,
        messages: list[ChatMessage],
        config: ModelConfig,
        model: str,
    ) -> AsyncIterator[str]:
        """Un intento de streaming."""


class OpenAIClient(BaseLLMClient):
    provider_name = "openai"

    def __init__(self, api_key: str, model: str) -> None:
        super().__init__(api_key, model)
        self._sdk = AsyncOpenAI(api_key=api_key)

    async def _generate_once(
        self,
        messages: list[ChatMessage],
        config: ModelConfig,
        model: str,
    ) -> ModelResponse:
        response = await self._sdk.chat.completions.create(
            model=model,
            messages=_openai_messages(messages),
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        content = response.choices[0].message.content
        if not isinstance(content, str):
            content = ""
        return ModelResponse(
            content=content,
            model=model,
            provider=self.provider_name,
            error=None,
        )

    async def _stream_once(
        self,
        messages: list[ChatMessage],
        config: ModelConfig,
        model: str,
    ) -> AsyncIterator[str]:
        response = await self._sdk.chat.completions.create(
            model=model,
            messages=_openai_messages(messages),
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            stream=True,
        )
        async for chunk in response:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            content = choices[0].delta.content
            if isinstance(content, str) and content != "":
                yield content


class AnthropicClient(BaseLLMClient):
    provider_name = "anthropic"

    def __init__(self, api_key: str, model: str) -> None:
        super().__init__(api_key, model)
        self._sdk = AsyncAnthropic(api_key=api_key)

    async def _generate_once(
        self,
        messages: list[ChatMessage],
        config: ModelConfig,
        model: str,
    ) -> ModelResponse:
        response = await self._sdk.messages.create(
            **_anthropic_kwargs(messages, config, model)
        )
        return ModelResponse(
            content=_anthropic_text(response),
            model=model,
            provider=self.provider_name,
            error=None,
        )

    async def _stream_once(
        self,
        messages: list[ChatMessage],
        config: ModelConfig,
        model: str,
    ) -> AsyncIterator[str]:
        async with self._sdk.messages.stream(
            **_anthropic_kwargs(messages, config, model)
        ) as stream:
            async for text in stream.text_stream:
                if isinstance(text, str) and text != "":
                    yield text
