"""Fachada que elige el cliente según la configuración."""

import logging
import os
from collections.abc import AsyncIterator

from clients import AnthropicClient, BaseLLMClient, OpenAIClient
from schemas import ChatMessage, LLMSettings, ModelConfig, ModelResponse, Provider

logger = logging.getLogger(__name__)


def settings_from_env() -> LLMSettings:
    raw_provider = os.environ.get("LLM_PROVIDER", "openai").strip() or "openai"
    return LLMSettings(
        provider=Provider(raw_provider),
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )


class AsyncLLMManager:
    """Delega generate y stream al cliente del proveedor configurado."""

    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings
        api_key = settings.active_api_key.get_secret_value()
        model = settings.active_model
        if settings.provider == Provider.OPENAI:
            self._client: BaseLLMClient = OpenAIClient(api_key=api_key, model=model)
        else:
            self._client = AnthropicClient(api_key=api_key, model=model)
        logger.info("cliente creado para %s", settings.provider.value)

    async def generate(
        self,
        messages: list[ChatMessage],
        config: ModelConfig | None = None,
    ) -> ModelResponse:
        return await self._client.generate(messages, config)

    async def stream(
        self,
        messages: list[ChatMessage],
        config: ModelConfig | None = None,
    ) -> AsyncIterator[str]:
        async for token in self._client.stream(messages, config):
            yield token
