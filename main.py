"""Ejemplo: una respuesta completa y después el mismo prompt en streaming."""

import asyncio
import logging
import os
from pathlib import Path

from async_llm_manager import AsyncLLMManager, settings_from_env
from schemas import ChatMessage


def load_dotenv(path: Path | None = None) -> None:
    """Lee un .env simple. No pisa variables que ya vienen del entorno."""
    env_path = path if path is not None else Path.cwd() / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_dotenv()
    manager = AsyncLLMManager(settings_from_env())
    messages = [ChatMessage(role="user", content="¿Qué es la entropía?")]

    response = await manager.generate(messages)
    if response.error:
        print(response.error)
    else:
        print(response.content)

    async for token in manager.stream(messages):
        print(token, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
