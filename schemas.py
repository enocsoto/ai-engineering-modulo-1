"""Esquemas de entrada, configuración y respuesta."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=32_000)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    model: str | None = None


class ModelResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str
    model: str
    provider: str
    error: str | None = None


class LLMSettings(BaseModel):
    """Proveedor activo y las claves que puede usar."""

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
        self.active_api_key
        return self

    @property
    def active_api_key(self) -> SecretStr:
        key = (
            self.openai_api_key
            if self.provider == Provider.OPENAI
            else self.anthropic_api_key
        )
        if key is None or key.get_secret_value().strip() == "":
            raise ValueError(
                f"Falta la clave de API del proveedor '{self.provider.value}'."
            )
        return key

    @property
    def active_model(self) -> str:
        if self.provider == Provider.OPENAI:
            return self.openai_model
        return self.anthropic_model
