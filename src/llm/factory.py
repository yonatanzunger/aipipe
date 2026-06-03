"""Factory functions for creating LLM clients and chat backends."""

from __future__ import annotations

from llm.chat import ChatBackend, ClientChatBackend
from llm.client import LLMClient
from llm.config import LLMConfig
from llm.impl.anthropic import _ANTHROPIC_TIER_DEFAULTS, AnthropicClient
from llm.impl.azure_inference import _AZURE_TIER_DEFAULTS, AzureInferenceClient
from llm.impl.claude_sdk import SdkChatBackend
from llm.impl.gemini import _GEMINI_TIER_DEFAULTS, GeminiClient
from llm.impl.github_copilot import _GITHUB_TIER_DEFAULTS, CopilotChatBackend
from llm.impl.openai import _OPENAI_TIER_DEFAULTS, OpenAIClient

_TIER_DEFAULTS_BY_PROVIDER: dict[str, dict[str, str]] = {
    "anthropic": _ANTHROPIC_TIER_DEFAULTS,
    "azure": _AZURE_TIER_DEFAULTS,
    "openai": _OPENAI_TIER_DEFAULTS,
    "github": _GITHUB_TIER_DEFAULTS,
    "gemini": _GEMINI_TIER_DEFAULTS,
}


def get_provider_tier_defaults(
    provider: str,
    auth_mode: str | None = None,
) -> dict[str, str]:
    """Return TIER_DEFAULTS for a provider without instantiating a backend.

    For providers with auth modes that use a different backend (e.g.
    Anthropic's ``claude_sdk`` mode uses :class:`SdkChatBackend` which
    has its own tier defaults), pass *auth_mode* to get the right set.
    """
    return _TIER_DEFAULTS_BY_PROVIDER.get(provider, {})


def create_client(config: LLMConfig) -> LLMClient:
    """Create a low-level LLM client for the given configuration.

    Args:
        config: Resolved :class:`LLMConfig` specifying the provider and
            credentials.

    Returns:
        An :class:`LLMClient` instance.

    Raises:
        ValueError: If the provider is not recognized or the auth mode
            does not support a low-level client (e.g. ``claude_sdk``).
    """
    if config.provider == "anthropic":
        if config.auth_mode == "claude_sdk":
            raise ValueError(
                "The claude_sdk auth mode does not use a low-level LLMClient. "
                "Use create_chat_backend() instead."
            )
        assert config.api_key is not None, "Anthropic provider requires an API key"
        return AnthropicClient(api_key=config.api_key)
    if config.provider == "azure":
        assert config.endpoint is not None, "Azure provider requires an endpoint"
        return AzureInferenceClient(
            endpoint=config.endpoint,
            api_key=config.api_key,
            auth_mode=config.auth_mode or "default",
            tenant_id=config.tenant_id,
        )
    if config.provider == "openai":
        assert config.api_key is not None, "OpenAI provider requires an API key"
        return OpenAIClient(api_key=config.api_key)
    if config.provider == "gemini":
        assert config.api_key is not None, "Gemini provider requires an API key"
        return GeminiClient(api_key=config.api_key)
    if config.provider == "github":
        raise ValueError(
            "The github provider uses the Copilot SDK backend, not a "
            "low-level LLMClient. Use create_chat_backend() instead."
        )
    raise ValueError(
        f"Unknown LLM provider: {config.provider!r}. "
        f"Supported providers: 'anthropic', 'azure', 'gemini', 'github', 'openai'"
    )


def create_chat_backend(config: LLMConfig) -> ChatBackend:
    """Create a high-level chat backend for the given configuration.

    For API-backed providers (anthropic, openai, azure, gemini), creates a
    :class:`ClientChatBackend` wrapping the appropriate :class:`LLMClient`.
    For Anthropic with ``claude_sdk`` auth mode, creates a
    :class:`SdkChatBackend` that wraps the Claude Code agent runtime; for
    ``github`` it creates a :class:`CopilotChatBackend`.

    Args:
        config: Resolved :class:`LLMConfig` specifying the provider,
            model, and credentials.

    Returns:
        A :class:`ChatBackend` instance.

    Raises:
        ValueError: If the provider is not recognized.
    """
    # The claude_sdk auth mode uses a fundamentally different backend
    # that wraps the Claude Code agent runtime rather than a raw API.
    if config.provider == "anthropic" and config.auth_mode == "claude_sdk":
        return SdkChatBackend()

    # The github provider uses the Copilot SDK agent runtime.
    if config.provider == "github":
        return CopilotChatBackend()

    # All other provider+auth combinations use ClientChatBackend.
    client = create_client(config)
    return ClientChatBackend(client, tiers=config.tiers)
