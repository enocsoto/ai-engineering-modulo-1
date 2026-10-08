# Módulo 1 — AsyncLLMManager

Gestor asíncrono que elige OpenAI o Anthropic desde la configuración, transmite los tokens en el momento y no se cae si la red falla o el proveedor responde con rate limit.

## Cómo correrlo

Requiere Python 3.12 o superior.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e .
cp .env.example .env
```

Completa `.env` con la clave del proveedor y ejecuta:

```bash
set -a && source .env && set +a
.venv/bin/python async_llm_manager.py
```

Los tokens salen a medida que llegan. La API key no se imprime: queda en `SecretStr`.

## Qué cubre

| Criterio | Dónde |
| --- | --- |
| Abstracción de proveedores | `AsyncLLMManager` arma el cliente según `LLMSettings.provider` |
| Streaming asíncrono | `stream()` es un generador async y entrega cada token |
| Validación con Pydantic | `GenerationParams` limita prompt, `temperature` (0 a 2) y `max_tokens` (1 a 4096) |
| Resiliencia | Reintenta fallos de red y rate limit. Si se agotan, registra el error y sigue |

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```
