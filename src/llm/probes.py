"""Live connectivity probes for LLM providers.

A probe makes a trivial real API/SDK round-trip to confirm that a provider's
credentials actually work — invaluable for debugging auth problems at setup
time. Ported (and trimmed) from clarity-agent's setup doctor.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass


@dataclass
class ProbeResult:
    """Outcome of a live provider probe."""

    ok: bool
    message: str
    hint: str | None = None


def _classify_error(error: Exception, provider: str) -> str:
    """Return a targeted, human-readable fix hint for common backend errors."""
    msg = str(error).lower()
    if "auth" in msg or "api key" in msg or "401" in msg or "403" in msg:
        return f"Your API key for {provider} may be invalid or expired."
    if "connection" in msg or "timeout" in msg or "resolve" in msg:
        return f"Network error connecting to {provider}. Check your internet connection."
    if "rate" in msg or "429" in msg:
        return f"Rate-limited by {provider}. Try again in a few minutes."
    if "billing" in msg or "payment" in msg or "quota" in msg:
        return f"Billing issue with {provider}. Check your account status."
    return f"{type(error).__name__}: {str(error)[:200]}"


def _probe_api(provider: str) -> ProbeResult:
    """Probe an API-based provider with a trivial create_message call."""
    from llm.config import _PROVIDERS, LLMConfig
    from llm.factory import get_provider_tier_defaults

    info = _PROVIDERS[provider]
    api_key: str | None = None
    for mode in info["auth_modes"]:
        ev = mode.get("env_var")
        if ev:
            api_key = os.environ.get(ev)
            if api_key:
                break
    endpoint = (
        os.environ.get(info["endpoint_env_var"])
        if info.get("endpoint_env_var") else None
    )
    default_model = get_provider_tier_defaults(provider).get("default", "unknown")
    # Honour an explicit deployment/model override (e.g. Azure deployment name).
    default_model = os.environ.get("AIPIPE_MODEL_DEFAULT", default_model)

    config = LLMConfig(
        provider=provider,
        api_key=api_key,
        endpoint=endpoint,
        tiers={"default": default_model},
    )
    client = config.create_client()
    asyncio.run(client.create_message(
        messages=[{"role": "user", "content": "Say ok"}],
        model=config.tiers.get("default", "unknown"),
        max_tokens=64,
        system="Respond with exactly: ok",
    ))
    # Any non-error round-trip means the connection works; the exact reply
    # doesn't matter (reasoning models may not echo "ok").
    return ProbeResult(ok=True, message=f"Provider {provider} responded successfully")


def _probe_sdk() -> ProbeResult:
    """Probe the Claude Agent SDK by spinning up an SdkChatBackend."""
    from llm.impl.claude_sdk import SdkChatBackend

    backend = SdkChatBackend()
    # The SDK can deliver a non-empty reply *and* then report an error (e.g. the
    # CLI exiting with an auth failure after streaming some text). It surfaces
    # that via on_warning rather than raising, so a non-empty reply alone is not
    # proof of success — capture warnings and treat them as failure.
    warnings: list[str] = []
    backend.on_warning = warnings.append
    try:
        reply = backend.chat("Say ok", system_prompt="Respond with exactly: ok", model="fast")
    finally:
        backend.disconnect()
    if warnings:
        return ProbeResult(ok=False, message=warnings[-1])
    if reply and reply.strip():
        return ProbeResult(ok=True, message="Claude SDK responded successfully")
    return ProbeResult(ok=False, message="Claude SDK returned an empty response")


def _probe_copilot() -> ProbeResult:
    """Probe the GitHub Copilot SDK by spinning up a CopilotChatBackend."""
    from llm.impl.github_copilot import CopilotChatBackend

    backend = CopilotChatBackend()
    try:
        reply = backend.chat("Say ok", system_prompt="Respond with exactly: ok", model="fast")
    finally:
        backend.disconnect()
    if reply and reply.strip():
        return ProbeResult(ok=True, message="GitHub Copilot responded successfully")
    return ProbeResult(ok=False, message="GitHub Copilot returned an empty response")


def probe(provider: str, auth_mode: str | None = None) -> ProbeResult:
    """Make a live round-trip to *provider* and report whether it works.

    Reads credentials from the environment (and thus from any ``Settings``
    values already injected into ``os.environ``). Never raises — failures are
    returned as ``ProbeResult(ok=False, ...)`` with a classified hint.
    """
    try:
        if provider == "anthropic" and auth_mode == "claude_sdk":
            return _probe_sdk()
        if provider == "github":
            return _probe_copilot()
        return _probe_api(provider)
    except Exception as e:  # noqa: BLE001 — probes intentionally catch everything
        return ProbeResult(ok=False, message=str(e), hint=_classify_error(e, provider))
