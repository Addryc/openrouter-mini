"""Minimal, shared OpenRouter chat-completions adapter (httpx, no SDK).

Public surface: a callable ``OpenRouterClient`` plus the ``Prompt`` / ``Usage`` /
``OpenRouterConfig`` types and typed errors. See ADR 0005 in solo-mud-platform.
"""

from openrouter_mini.client import (
    DEFAULT_MODEL,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    OPENROUTER_API_KEY_ENV,
    OPENROUTER_CHAT_COMPLETIONS_URL,
    OPENROUTER_MODEL_ENV,
    OpenRouterClient,
    OpenRouterConfig,
    OpenRouterConfigurationError,
    OpenRouterError,
    OpenRouterRequestError,
    OpenRouterResponseError,
    Prompt,
    Usage,
    build_client,
    load_client,
    load_config,
)

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_REQUEST_TIMEOUT_SECONDS",
    "OPENROUTER_API_KEY_ENV",
    "OPENROUTER_CHAT_COMPLETIONS_URL",
    "OPENROUTER_MODEL_ENV",
    "OpenRouterClient",
    "OpenRouterConfig",
    "OpenRouterConfigurationError",
    "OpenRouterError",
    "OpenRouterRequestError",
    "OpenRouterResponseError",
    "Prompt",
    "Usage",
    "build_client",
    "load_client",
    "load_config",
]

__version__ = "0.1.0"
