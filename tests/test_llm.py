import argparse

import keyring
import pytest

from llm import complete
from llm.chat import ClientChatBackend
from llm.client import LLMClient
from llm.config import LLMConfig, _auto_detect_provider
from llm.keyring_backend import LocalKeyring
from llm.probes import _classify_error
from llm.settings import Settings
from llm.types import LLMResponse, TextBlock, TokenUsage, ToolUseBlock


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Isolate Settings storage: tmp data dir + a local-file keyring, no real
    keychain, no ambient provider env vars."""
    monkeypatch.setenv("AIPIPE_DATA_DIR", str(tmp_path))
    keyring.set_keyring(LocalKeyring(path=tmp_path / "secrets.json"))
    for var in (
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "AZURE_AI_API_KEY",
        "AZURE_AI_ENDPOINT", "GEMINI_API_KEY", "GOOGLE_API_KEY", "GITHUB_TOKEN",
        "AIPIPE_LLM_PROVIDER", "AIPIPE_AUTH_MODE", "CLAUDECODE",
    ):
        monkeypatch.delenv(var, raising=False)
    Settings._reset()
    yield
    Settings._reset()


class FakeClient(LLMClient):
    TIER_DEFAULTS = {"default": "fake-1", "fast": "fake-fast"}

    async def _create_message(self, *, messages, model, max_tokens=4096, system=None, tools=None):
        return LLMResponse(content=[TextBlock(text=f"[{model}] {messages[-1]['content']}")])


# --------------------------------------------------------------------------- #
# types
# --------------------------------------------------------------------------- #


def test_llmresponse_text_and_tool_calls():
    r = LLMResponse(content=[
        TextBlock(text="hello"),
        ToolUseBlock(id="t1", name="Read", input={"file_path": "/x"}),
        TextBlock(text="world"),
    ])
    assert r.text == "hello\nworld"
    assert [t.name for t in r.tool_calls] == ["Read"]
    dicts = r.content_as_dicts
    assert dicts[1] == {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "/x"}}


def test_token_usage_add():
    total = TokenUsage(1, 2) + TokenUsage(10, 20)
    assert (total.input_tokens, total.output_tokens, total.total_tokens) == (11, 22, 33)


# --------------------------------------------------------------------------- #
# chat / complete
# --------------------------------------------------------------------------- #


def test_complete_via_fake_backend():
    be = ClientChatBackend(FakeClient(), tiers={"default": "fake-1"})
    be.connect()
    try:
        out = complete("ping", model="default", backend=be)
    finally:
        be.disconnect()
    assert out == "[fake-1] ping"


def test_resolve_model_uses_tier_overrides():
    be = ClientChatBackend(FakeClient(), tiers={"default": "override-model"})
    assert be.resolve_model("default") == "override-model"
    assert be.resolve_model("fast") == "fake-fast"  # falls back to client default
    assert be.resolve_model("literal-model") == "literal-model"


def test_chat_passes_system_prompt_verbatim():
    be = ClientChatBackend(FakeClient(), tiers={"default": "m"})
    # _build_system_prompt is pure passthrough now (no clarity boilerplate).
    assert be._build_system_prompt("just this") == "just this"
    assert be._build_system_prompt(None) == ""


# --------------------------------------------------------------------------- #
# config / auto-detect
# --------------------------------------------------------------------------- #


def test_auto_detect_prefers_anthropic(isolated, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    assert _auto_detect_provider() == ("anthropic", "api_key")


def test_auto_detect_none_when_empty(isolated, monkeypatch):
    # No provider env vars and no gh CLI token available. get_gh_cli_token is
    # imported into llm.config's namespace, so patch it there.
    monkeypatch.setattr("llm.config.get_gh_cli_token", lambda *a, **k: None)
    assert _auto_detect_provider() is None


def test_config_create_resolves_and_persists(isolated, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-abc")
    ns = argparse.Namespace(provider=None, api_key=None, endpoint=None, model="claude-x",
                            model_deep=None, model_fast=None, auth_mode=None)
    cfg = LLMConfig.create(ns)
    assert cfg.provider == "anthropic" and cfg.auth_mode == "api_key"
    assert cfg.tiers["default"] == "claude-x"
    # Persisted to the isolated settings store.
    Settings._reset()
    assert Settings.current().provider == "anthropic"


# --------------------------------------------------------------------------- #
# settings
# --------------------------------------------------------------------------- #


def test_settings_roundtrip(isolated):
    s = Settings.current()
    s.set("ANTHROPIC_API_KEY", "sk-secret")
    s.set("AIPIPE_LLM_PROVIDER", "anthropic")
    s.model_default = "claude-x"
    s.save()

    Settings._reset()
    s2 = Settings.current()
    assert s2.provider == "anthropic"
    assert s2.anthropic_api_key == "sk-secret"
    assert s2.tier_overrides == {"default": "claude-x"}


def test_settings_set_rejects_unknown_key(isolated):
    with pytest.raises(KeyError):
        Settings.current().set("NOT_A_KEY", "x")


# --------------------------------------------------------------------------- #
# probes (error classification — no network)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("text,expect", [
    ("401 Unauthorized: bad api key", "invalid or expired"),
    ("Connection timeout while resolving host", "Network error"),
    ("429 rate limit exceeded", "Rate-limited"),
    ("payment required: quota exceeded", "Billing issue"),
])
def test_classify_error(text, expect):
    assert expect in _classify_error(Exception(text), "anthropic")
