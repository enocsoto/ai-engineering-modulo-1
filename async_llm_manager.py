"""Async manager that streams tokens from OpenAI or Anthropic."""

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from enum import Enum

from anthropic import APIConnectionError as AnthropicConnectionError
from anthropic import APITimeoutError as AnthropicTimeoutError
from anthropic import AsyncAnthropic
from anthropic import RateLimitError as AnthropicRateLimitError
from openai import APIConnectionError as OpenAIConnectionError
from openai import APITimeoutError as OpenAITimeoutError
from openai import AsyncOpenAI
from openai import RateLimitError as OpenAIRateLimitError
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

logger = logging.getLogger(__name__)

RETRYABLE = (
    OpenAIConnectionError,
    OpenAITimeoutError,
    OpenAIRateLimitError,
    AnthropicConnectionError,
    AnthropicTimeoutError,
    AnthropicRateLimitError,
)
MAX_ATTEMPTS = 3


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class LLMSettings(BaseModel):
    """Which provider to call, and the secret it needs."""

    model_config = ConfigDict(extra="forbid")

    provider: Provider
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-3-5-haiku-latest"

    @field_validator("openai_api_key", "anthropic_api_key", mode="before")
    @classmethod
    def blank_key_to_none(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, SecretStr):
            raw = value.get_secret_value()
        elif isinstance(value, str):
            raw = value
        else:
            return value
        if raw.strip() == "":
            return None
        return value

    @model_validator(mode="after")
    def selected_provider_has_key(self) -> "LLMSettings":
        self.key_for()
        return self

    def key_for(self) -> SecretStr:
        key = (
            self.openai_api_key
            if self.provider is Provider.OPENAI
            else self.anthropic_api_key
        )
        if key is None or key.get_secret_value().strip() == "":
            raise ValueError(
                f"API key for provider '{self.provider.value}' is missing or blank."
            )
        return key

    @property
    def model(self) -> str:
        if self.provider is Provider.OPENAI:
            return self.openai_model
        return self.anthropic_model


class GenerationParams(BaseModel):
    """Input bounds checked before any provider call."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=32_000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=4096)


class AsyncLLMManager:
    """Stream one configured provider. Network and rate-limit errors are retried."""

    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings
        secret = settings.key_for().get_secret_value()
        if settings.provider is Provider.OPENAI:
            self._client: AsyncOpenAI | AsyncAnthropic = AsyncOpenAI(api_key=secret)
        else:
            self._client = AsyncAnthropic(api_key=secret)
        logger.info("cliente creado para %s", settings.provider.value)

    async def stream(self, params: GenerationParams) -> AsyncIterator[str]:
        """Yield text tokens. Give up after retries without raising."""
        delay = 0.5
        for attempt in range(1, MAX_ATTEMPTS + 1):
            started = False
            try:
                async for token in self._stream_once(params):
                    started = True
                    yield token
                return
            except RETRYABLE as exc:
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

    async def _stream_once(self, params: GenerationParams) -> AsyncIterator[str]:
        if self._settings.provider is Provider.OPENAI:
            async for token in self._stream_openai(params):
                yield token
            return
        async for token in self._stream_anthropic(params):
            yield token

    async def _stream_openai(self, params: GenerationParams) -> AsyncIterator[str]:
        response = await self._client.chat.completions.create(
            model=self._settings.model,
            messages=[{"role": "user", "content": params.prompt}],
            temperature=params.temperature,
            max_tokens=params.max_tokens,
            stream=True,
        )
        async for chunk in response:
            content = chunk.choices[0].delta.content
            if isinstance(content, str) and content != "":
                yield content

    async def _stream_anthropic(self, params: GenerationParams) -> AsyncIterator[str]:
        async with self._client.messages.stream(
            model=self._settings.model,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            messages=[{"role": "user", "content": params.prompt}],
        ) as stream:
            async for text in stream.text_stream:
                if isinstance(text, str) and text != "":
                    yield text


def settings_from_env() -> LLMSettings:
    return LLMSettings(
        provider=Provider(os.environ.get("LLM_PROVIDER", "openai")),
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    manager = AsyncLLMManager(settings_from_env())
    params = GenerationParams(prompt=os.environ.get("LLM_PROMPT", "Di hola en una frase."))
    async for token in manager.stream(params):
        print(token, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
