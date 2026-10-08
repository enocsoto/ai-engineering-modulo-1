# Módulo 1 — cliente asíncrono de LLM

Cliente que llama a OpenAI o Anthropic, en una sola respuesta o en streaming. Valida la entrada con Pydantic y reintenta fallos de red sin detener el programa.

## Cómo correrlo

Requiere Python 3.12.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e .
cp .env.example .env
```

Completa en `.env` la clave del proveedor elegido y ejecuta:

```bash
.venv/bin/python main.py
```

La pregunta fija es «¿Qué es la entropía?». Primero se imprime la respuesta completa. Después se imprimen los tokens a medida que llegan. Si el proveedor falla, el script muestra el error o lo registra y termina.

## Variables de entorno

| Variable | Uso |
| --- | --- |
| `LLM_PROVIDER` | `openai` (por defecto) o `anthropic` |
| `OPENAI_API_KEY` | Clave de OpenAI. Obligatoria si el proveedor es `openai` |
| `ANTHROPIC_API_KEY` | Clave de Anthropic. Obligatoria si el proveedor es `anthropic` |

Una clave en blanco o con solo espacios se trata como ausente. El valor no se imprime: queda en `SecretStr`.

`main.py` lee el `.env` del directorio actual y no pisa variables que ya estén exportadas.

## Dónde está cada criterio

| Criterio | Dónde |
| --- | --- |
| Esquemas Pydantic (`Provider`, `ChatMessage`, `ModelConfig`, `ModelResponse`, `LLMSettings`) | `schemas.py` |
| `BaseLLMClient`, `OpenAIClient` y `AnthropicClient` | `clients.py` |
| Modo normal `generate` y streaming `stream` | `clients.py`; la fachada los expone en `AsyncLLMManager` |
| Fachada `AsyncLLMManager` y lectura del entorno | `async_llm_manager.py` |
| Reintentos (conexión, timeout y rate limit; sin reintento si ya salió un token o si la clave es inválida) | `clients.py` |
| Script de ejemplo | `main.py` |

En `generate`, un fallo de red, timeout, rate limit o clave inválida vuelve como `ModelResponse` con `content` vacío y `error` en español. En `stream`, se reintenta solo antes del primer token; si los reintentos se agotan, se registra el fallo y el generador termina.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```
