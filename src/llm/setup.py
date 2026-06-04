"""Interactive setup and config management for the LLM backend.

This is the terminal replacement for clarity-agent's setup GUI: it renders the
same ``_PROVIDERS`` registry metadata (display names, auth modes, fields) as a
prompt-driven wizard, then makes a live API call to validate the credentials
before saving them via :class:`~llm.settings.Settings`.
"""

from __future__ import annotations

from getpass import getpass
from typing import Any

from llm.config import _PROVIDERS, get_auth_mode_info, get_auth_mode_names
from llm.probes import probe
from llm.settings import _SECRET_KEYS, Settings


def _choose(prompt: str, options: list[tuple[str, str]]) -> str:
    """Prompt the user to pick from ``options`` (value, label); return value."""
    for i, (_value, label) in enumerate(options, 1):
        print(f"  {i}. {label}")
    while True:
        raw = input(f"{prompt} [1-{len(options)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1][0]
        print("  Please enter a number from the list.")


def _collect_fields(fields: list[dict[str, Any]]) -> dict[str, str]:
    """Prompt for each setup field; mask secrets. Returns {key: value}."""
    values: dict[str, str] = {}
    for field in fields:
        key = field["key"]
        label = field.get("label", key)
        if field.get("help"):
            print(f"    ({field['help']})")
        placeholder = field.get("placeholder")
        suffix = f" [{placeholder}]" if placeholder else ""
        if field.get("secret"):
            value = getpass(f"  {label}{suffix}: ").strip()
        else:
            value = input(f"  {label}{suffix}: ").strip()
        if value:
            values[key] = value
        elif not field.get("optional"):
            print(f"    (no value entered for {label})")
    return values


def run_wizard(settings: Settings | None = None) -> bool:
    """Run the interactive setup wizard. Returns True if a working provider
    was configured and saved."""
    s = settings if settings is not None else Settings.current()

    print("Configure an LLM provider.\n")
    provider = _choose(
        "Provider",
        [(name, info["display_name"]) for name, info in _PROVIDERS.items()],
    )
    info = _PROVIDERS[provider]
    if info.get("setup_url"):
        print(f"  Setup help: {info['setup_url']}")

    modes = get_auth_mode_names(provider)
    if len(modes) == 1:
        auth_mode = modes[0]
    else:
        auth_mode = _choose(
            "\nAuthentication method",
            [(m, get_auth_mode_info(provider, m)["display_name"]) for m in modes],
        )
    mode_info = get_auth_mode_info(provider, auth_mode) or {}
    if mode_info.get("setup_help"):
        print(f"\n{mode_info['setup_help']}")
    if mode_info.get("setup_url"):
        print(f"Get credentials at: {mode_info['setup_url']}")

    print()
    fields = list(info.get("common_fields", [])) + list(mode_info.get("fields", []))
    credentials = _collect_fields(fields)

    # Persist selection + credentials.
    s.set("AIPIPE_LLM_PROVIDER", provider)
    s.set("AIPIPE_AUTH_MODE", auth_mode)
    s.provider_auth_modes[provider] = auth_mode
    for key, value in credentials.items():
        s.set(key, value)
    s.save()

    # Live validation.
    print("\nTesting the connection ...")
    result = probe(provider, auth_mode)
    if result.ok:
        print(f"  ✓ {result.message}")
        print(f"\nSaved. '{provider}' is ready to use.")
        return True
    print(f"  ✗ {result.message}")
    if result.hint:
        print(f"  Hint: {result.hint}")
    print("\nCredentials were saved but the test call failed; fix the issue and "
          "re-run setup or `config set`.")
    return False


def config_show(settings: Settings | None = None) -> None:
    """Print the current (non-secret) configuration and which secrets are set."""
    s = settings if settings is not None else Settings.current()
    print(f"provider:   {s.provider or '(unset)'}")
    print(f"auth_mode:  {s.auth_mode or '(unset)'}")
    for tier in ("model_default", "model_deep", "model_fast"):
        val = getattr(s, tier)
        if val:
            print(f"{tier}: {val}")
    print("secrets:")
    for attr, env_key in _SECRET_KEYS.items():
        marker = "set" if getattr(s, attr) else "—"
        print(f"  {env_key}: {marker}")
    print(f"settings file: {s.settings_path}")


def config_set(env_key: str, value: str, settings: Settings | None = None) -> None:
    """Set one config/credential value by its env-var name and save.

    An empty/whitespace value clears the setting (equivalent to ``unset``), so
    it never leaves an empty string masquerading as a real credential.
    """
    s = settings if settings is not None else Settings.current()
    if not value.strip():
        config_unset(env_key, settings=s)
        return
    s.set(env_key, value)
    s.save()
    print(f"Set {env_key}.")


def config_unset(env_key: str, settings: Settings | None = None) -> None:
    """Clear one config/credential value by its env-var name and save."""
    s = settings if settings is not None else Settings.current()
    s.set(env_key, None)
    s.save()
    print(f"Unset {env_key}.")
