"""A portable, provider-agnostic LLM backend.

Ported from clarity-agent. Exposes a uniform interface for talking to several
LLM providers (Anthropic, OpenAI, Azure, Gemini, plus the Claude Agent SDK and
GitHub Copilot SDK) and switching between them via configuration.

Quick start::

    from llm import complete
    print(complete("Say hello in one word."))

Provider/credentials are resolved from the environment and the persisted
``Settings`` store (see ``llm.settings``); configure them with the ``llm.setup``
wizard or by setting e.g. ``ANTHROPIC_API_KEY``.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from llm.client import LLMClient
from llm.types import (
    CompactionInfo,
    LLMAuthExpiredError,
    LLMResponse,
    TextBlock,
    TokenUsage,
    ToolUseBlock,
)

if TYPE_CHECKING:
    from llm.chat import ChatBackend
    from llm.settings import Settings

__all__ = [
    "CompactionInfo",
    "LLMAuthExpiredError",
    "LLMClient",
    "LLMResponse",
    "TextBlock",
    "TokenUsage",
    "ToolUseBlock",
    "complete",
    "create_backend",
]


def create_backend(
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    endpoint: str | None = None,
    auth_mode: str | None = None,
    settings: "Settings | None" = None,
    connect: bool = True,
) -> "ChatBackend":
    """Build a connected :class:`~llm.chat.ChatBackend` for any provider.

    Provider/auth/credentials default to auto-detection from the environment
    and the persisted ``Settings`` store; pass any field to override. The
    returned backend is connected (ready for :meth:`chat`) unless
    ``connect=False``; remember to call ``.disconnect()`` when done.
    """
    from llm.config import LLMConfig

    ns = argparse.Namespace(
        provider=provider, api_key=api_key, endpoint=endpoint,
        model=model, model_deep=None, model_fast=None, auth_mode=auth_mode,
    )
    config = LLMConfig.create(ns, settings=settings)
    backend = config.create_chat_backend()
    if connect:
        backend.connect()
    return backend


def complete(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    backend: "ChatBackend | None" = None,
) -> str:
    """Synchronous one-shot completion across any provider.

    If ``backend`` is given it is used as-is (and left connected); otherwise a
    backend is created from the ambient configuration, used once, and
    disconnected.
    """
    own = backend is None
    be = backend if backend is not None else create_backend(model=model)
    try:
        return be.chat(prompt, system_prompt=system, model=model)
    finally:
        if own:
            be.disconnect()
