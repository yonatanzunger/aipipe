# Implementation plans

Detailed plans for the four items in [TODO.md](TODO.md). Code blocks are
proposals/sketches, not final.

Source facts these plans are built on:
- clarity-agent's LLM backend lives in `src/clarity_agent/llm/`:
  `types.py`, `client.py` (async `LLMClient`), `chat.py` (`ChatBackend` +
  `ClientChatBackend`), `config.py`, `factory.py`, and `impl/{anthropic,openai,
  azure_inference,gemini,claude_sdk,github_copilot}.py`. The core
  (`types`, `client`, `impl/*`) has **no** clarity-agent imports. Couplings to
  strip live in `config.py`/`chat.py`: `app_paths.protocol_dir`,
  `settings.Settings.current()`, `process_registry`, optional `transcript`,
  and SDK-only `ai_actions.format_tools_as_cli`.
- `pyppin.os.bulk_import.bulk_import(path, recursive=, exclude=, root=, visitor=)`
  imports all `.py` files under a dir; `pyppin.base.import_file.import_file` does
  one file. `pyppin.util.retry` is an exponential-backoff helper.
- Current `dag.py`: `provider` decorator → `Providers.add(func)` →
  `Provider.build(func)` introspects the signature (return = `provides`, params =
  `requires`/`optionally_requires`). `make()` uses the module-global
  `_providers`. `Provider.call(**kwargs)` filters kwargs to its declared params
  and calls `self.func(**args)`.

---

## Item 1 — Port the full LLM backend (all providers) into a portable library

**Goal:** a self-contained `aipipe` LLM backend with no clarity-agent imports,
supporting **all of clarity-agent's providers** so we can switch providers with
a flag/env-var (the high-value feature), and exposing a dead-simple "prompt in,
text out" call for `LLMStage`.

**Where it lives:** new **top-level** package `src/llm/` (importable as `llm`),
*not* under `dag` — it's a general-purpose backend we may reuse elsewhere in
this package, independent of the DAG machinery. Update the wheel config to ship
both packages: `packages = ["src/dag", "src/llm"]`. `dag` may depend on `llm`,
never the reverse.

**The six providers come in two flavors** (important for the design):
- **`LLMClient` providers** (low-level async `create_message`):
  `impl/anthropic.py`, `impl/openai.py`, `impl/azure_inference.py`,
  `impl/gemini.py`. Each depends only on its own SDK, lazily imported.
- **`ChatBackend` providers** (full chat backends, not low-level clients):
  `impl/claude_sdk.py` (`SdkChatBackend`), `impl/github_copilot.py`
  (`CopilotChatBackend`).

These are unified by `chat.py`'s `ClientChatBackend`, which wraps *any*
`LLMClient` as a `ChatBackend`. So **`ChatBackend` is the single uniform surface
across all six providers** — `backend.chat(prompt, system_prompt=...)` returns
text everywhere. We center the port on that; it's exactly what makes provider
switching seamless.

**Files to port** (copy into `src/llm/`, then decouple — see checklist):
- `types.py` — verbatim (zero couplings).
- `client.py` — abstract async `LLMClient`. Verbatim.
- `chat.py` — `ChatBackend` + `ClientChatBackend`. **Needs decoupling.**
- `config.py` — provider registry, env auto-detect, `LLMConfig`,
  `add_arguments`/`create`. **Needs decoupling.** This is the heart of provider
  switching.
- `factory.py` — `create_client` / `create_chat_backend` /
  `get_provider_tier_defaults`. Light decoupling (transcript type only).
- `impl/anthropic.py`, `impl/openai.py`, `impl/azure_inference.py`,
  `impl/gemini.py` — verbatim (SDK-only deps).
- `impl/claude_sdk.py`, `impl/github_copilot.py` — needs the `ai_actions`/
  `app_paths` decoupling (below).
- `impl/_openai_compat.py` — shared helper for the OpenAI-compatible backends.

**Decoupling checklist** (the only clarity-agent couplings, all isolated):
- `app_paths.protocol_dir(project_dir)` (in `chat.py`, `claude_sdk.py`,
  `github_copilot.py`): used only to build a default system prompt. aipipe gets
  its system prompt from the `LLMStage` (front-matter `system`), so replace
  path-based system-prompt construction with the **passed-in `system_prompt`**;
  drop the path lookups.
- `settings.Settings.current()` (in `chat.py`, `config.py`): **port `Settings`
  itself** (see "Settings / credential management" below) rather than just
  stripping it. We need a real place to configure and persist LLM keys for the
  CLI; the explicit `LLMConfig` fields are the in-memory view, but `Settings` is
  the persisted store behind them.
- `process_registry.get_default_process_tiers()` (in `config.py`): drop the
  per-process tier mapping; keep the static per-provider `TIER_DEFAULTS` baked
  into each `impl/`.
- `transcript.Transcript` (TYPE_CHECKING-only, already `... | None`): keep
  optional, pass `None`. Compaction degrades gracefully without it.
- `ai_actions.format_tools_as_cli()` (in `claude_sdk.py` only): localize that
  helper into the SDK impl (it's small) or stub it until tools are needed.

**Public surface in `llm/__init__.py`:**

```python
def create_backend(*, provider: str | None = None, model: str | None = None,
                    api_key: str | None = None, endpoint: str | None = None,
                    auth_mode: str | None = None) -> ChatBackend:
    """Build a ChatBackend for any provider. provider/auth default from env via
    config's auto-detect (ANTHROPIC_API_KEY -> anthropic, OPENAI_API_KEY ->
    openai, AZURE_AI_ENDPOINT -> azure, GEMINI/GOOGLE_API_KEY -> gemini,
    GITHUB_TOKEN -> github, CLAUDECODE -> claude_sdk)."""

def complete(prompt: str, *, system: str | None = None, model: str | None = None,
             backend: ChatBackend | None = None) -> str:
    """Synchronous one-shot completion across any provider."""
    backend = backend or create_backend(model=model)
    return backend.chat(prompt, system_prompt=system, model=model)
```

`create_backend` is a thin wrapper over the ported
`LLMConfig` + `factory.create_chat_backend` (which already returns the right
`ChatBackend` per provider, incl. `ClientChatBackend` for the four API-key
clients). `complete()` uses the uniform `ChatBackend.chat()` — sync already, so
no asyncio plumbing at the call site. Wrap transient failures with
`pyppin.util.retry`.

`create_backend`/`complete` also take a `verbose: int = 0` that the backend
honors for debug output — see the reserved-resource convention below. This is
how the backend plugs into the uniform debug API.

**Provider switching in the CLI (item 4):** reuse `LLMConfig.add_arguments()` so
`--provider/--model/--api-key/--endpoint/--auth-mode/--model-deep/--model-fast`
land on the same parser, and `LLMConfig.create(args)` builds the config. One
shared backend is then passed to every stage.

**Dependencies — per-provider optional extras** (keep DAG core light, install
only what you switch to):

```toml
[project.optional-dependencies]
anthropic  = ["anthropic>=0.40"]
openai     = ["openai>=1.40"]
azure      = ["azure-ai-inference>=1.0", "azure-identity>=1.17"]
gemini     = ["google-genai>=0.3"]
copilot    = ["copilot"]            # confirm the real dist name
claude-sdk = ["claude-agent-sdk"]
llm-all    = ["aipipe[anthropic,openai,azure,gemini]"]
```

SDK imports are already lazy in each `impl/`, so a missing extra only errors
when that provider is actually selected.

**Risks / decisions:**
- Bigger port than a single-provider slice, but the providers are modular and
  the couplings are few and isolated (checklist above). Budget most of the time
  for `config.py`/`chat.py` decoupling, not the impls.
- Model/tier naming: keep clarity's tier aliases ("deep"/"fast") since
  `add_arguments` already exposes `--model-deep/--model-fast` and they're handy;
  `complete()` also accepts a literal model string.
- Confirm exact dist names/versions for the Azure, Gemini, and Copilot SDKs when
  wiring `pyproject.toml`.

### Item 1 — LOCKED DECISIONS (agreed 2026-06-03)

Package lives at top-level `src/llm/` (importable `llm`); `dag` may depend on it,
never the reverse. Coupling audit: most of clarity's coupling is mechanical or
already-optional (`transcript=None` cleanly disables all compaction in every
backend; `Transcript` hints are `TYPE_CHECKING`-only; API-key clients + claude_sdk
already lazy-import). The real decisions:

- **A. System prompt / paths (fixes chat.py + claude_sdk + github_copilot at
  once).** Backends do **no** prompt construction — `chat(system_prompt=…)` is
  passed through verbatim (pure passthrough, no base prompt). Drop
  `project_dir`/`clarity_agent_dir`/`protocol_dir` from all three constructors.
  This removes the entire `app_paths` coupling.
- **B. Config + credential layer.**
  - Port `Settings` as the persisted store; retarget storage to an aipipe data
    dir (platformdirs, e.g. `~/.config/aipipe/`); rename keyring `SERVICE`
    `"clarity"→"aipipe"`; keep precedence `settings.json < .env < keyring < env`.
  - **Secrets via keyring** (+ `python-dotenv`) — keep both.
  - Rename clarity-specific field keys (`CLARITY_MODEL_DEFAULT`,
    `CLARITY_TENANT_ID`, `CLARITY_LLM_PROVIDER`, …) → `AIPIPE_*`; keep standard
    SDK names (`ANTHROPIC_API_KEY`, …).
  - **Drop `process_registry`** entirely (no "processes" in aipipe): remove
    `resolve(process_name)`, `_DEFAULT_PROCESS_TIERS`, `process_overrides`.
  - **Keep tiers** `default`/`deep`/`fast` + `--model`/`--model-deep`/`--model-fast`.
  - Keep the `LLMConfig.create()` auto-detect facade; make `Settings` injectable/
    optional; **keep write-back** of resolved provider/auth.
- **C. Tools in the SDK backend.** Don't port `ai_actions`; stub
  `format_tools_as_cli → ""`. Keep the `ChatBackend` tool-loop interface, ship
  the SDK backend tools-unsupported for now (LLMStage needs text only).
- **D. Lazy imports / extras.** All six impls gate their SDK behind a lazy import
  + optional extra; fix `github_copilot`'s top-level `from copilot import …` to be
  lazy. Remove copilot's dead `token` param.
- **E. Packaging.** `azure-ai-inference` is prerelease-only (`1.0.0b9`) → extra
  needs prerelease opt-in; Gemini dist is `google-genai` (confirm version).

**Settings creation (replacing the GUI).** The `_PROVIDERS` registry already *is*
the setup-form schema (display names, auth modes, `fields[{key,label,secret,
placeholder,help}]`, `setup_url`). The clarity GUI was just a renderer + a
`Settings.save()`. Port three creation paths:
  1. **Env vars** — already highest precedence; zero setup (dev/CI).
  2. **Write-back** — `--provider/--api-key` on a normal run persists (decision B).
  3. **Terminal wizard** — `config`/`--setup` subcommand that walks `_PROVIDERS`,
     prompts per field (mask secrets via `getpass`), then **does a live API call**
     to validate before saving (port `setup/doctor.py`'s `_probe_api`/`_probe_sdk`/
     `_probe_copilot` + `_classify_error` — the live check proved very useful for
     debugging auth). Also `config show` / `config set KEY VALUE` for scripting.
  First-run, no creds: TTY → offer the wizard; non-TTY → print `setup_help` +
  `setup_url` and exit non-zero.

---

## Item 2 — `LLMStage` class

**Goal:** turn a Markdown template with `{{substitutions}}` into a DAG stage
whose required resources are the substitution variables and whose provided
resource is the LLM output.

**Where:** new module `src/dag/llm_stage.py` (imports the top-level `llm`
package — `from llm import complete, ChatBackend`). The stage is DAG-specific so
it stays under `dag`; only the provider-agnostic backend lives in `llm`.

**Shape:**

```python
_VAR = re.compile(r"{{\s*(\w+)\s*}}")

class LLMStage:
    def __init__(self, name: str, template: str, *,
                 backend: LLMClient | None = None,
                 model: str | None = None,
                 system: str | None = None):
        self.name = name                      # resource this stage provides
        self.template = template
        self.variables = list(dict.fromkeys(_VAR.findall(template)))  # required, de-duped, ordered
        self.backend = backend
        self.model = model
        self.system = system

    def render(self, **kwargs: Any) -> str:
        return _VAR.sub(lambda m: str(kwargs[m.group(1)]), self.template)

    def __call__(self, **kwargs: Any) -> str:
        return complete(self.render(**kwargs), system=self.system,
                        model=self.model, client=self.backend)

    def as_provider(self) -> Provider:
        return Provider(
            func=self,                        # __call__ is the provider body
            name=self.name,
            provides=str,                     # LLM output is text
            requires={v: Any for v in self.variables},   # see note below
            optionally_requires={},
        )
```

**Integration change in `dag.py`** (small, enables non-function providers):

```python
class Providers:
    def add(self, func):                      # unchanged entry point
        self.add_provider(Provider.build(func))

    def add_provider(self, p: Provider) -> None:
        self._declare_provided(p.name, p.provides)
        for n, a in p.requires.items():           self._declare_required(p.name, n, a)
        for n, a in p.optionally_requires.items(): self._declare_required(p.name, n, a)
        self.providers[p.name] = p
```

`Provider.call` already works unchanged: `func=self` (the stage), it filters
kwargs to `requires` and calls `stage(**vars)`.

**Decision (CONFIRMED) — model selection via a common resource:** every
`LLMStage` consumes the reserved resource `model` (a `str`) as an *optional*
requirement, so a single `--model` on the CLI sets the model for all stages at
once. Effective model precedence: front-matter `model` (per-stage override) >
the `model` resource > backend default. Implementation: add `model` to the
stage's `optionally_requires`, and in `__call__` resolve
`self.model or kwargs.get("model")`. See the reserved-resource convention below.

**Decision (CONFIRMED) — type of the template variables:** require them as
`Any` (not `str`). We stringify whatever is passed, so a `dict`/`int` producer
feeding a `{{var}}` is fine at runtime; typing them `str` would make the new
directional type-check reject a non-str producer. `Any` on the required side
short-circuits `_check_compatible` to "compatible". (If we ever want strictness,
make it configurable per-var.)

**Tests to add:** variable discovery (incl. de-dup and whitespace
`{{ x }}`), `render()` substitution, `as_provider()` wiring (a fake/stub backend
returning a canned string so `make()` runs end-to-end without network).

---

## Item 3 — Factory: `.md` file → `LLMStage`

**Goal:** read a Markdown file and produce the corresponding `LLMStage`.

**Where:** `src/dag/llm_stage.py`, alongside the class.

```python
def stage_from_file(path: str | Path, *, backend=None, model=None) -> LLMStage:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    name, system = _parse_front_matter(text)   # optional; see below
    return LLMStage(name=name or path.stem, template=text,
                    backend=backend, model=model, system=system)
```

**Decisions:**
- **Resource name** defaults to the filename stem (`summary.md` → provides
  `summary`). Allow override via front-matter.
- **Front-matter (optional, recommended):** support a leading `---` YAML/TOML
  block for `name`, `model`, `system`. If we want zero new deps, parse a tiny
  `key: value` block by hand rather than pulling in PyYAML. Strip it from the
  template body before substitution. Start without front-matter if we want to
  ship faster — filename-as-name covers the common case.
- Keep the factory free of registration side-effects (returns a stage); the
  loader in item 4 decides what registry to add it to.

**Tests:** a temp `.md` file → correct `name`, `variables`, and rendered output
(stub backend).

---

## Item 4 — `__main__`: bulk-load a dag dir, dynamic flags, `make()`

**Goal:** point the CLI at a directory of providers (`.py` `@provider`
functions and/or `.md` LLM stages), expose every resource as an optional
`--flag`, take target(s), and run `make()`.

**Where:** `src/dag/__main__.py` (so `python -m dag ...` works).

**Two-pass argparse** ("re-run the argument parser"):

```python
def main(argv=None):
    # Pass 1: just enough to find the dag dir.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--dagdir", required=True, type=Path)
    known, _ = pre.parse_known_args(argv)

    # Load providers from the dir.
    load_dag_dir(known.dagdir)          # see below; registers into the global _providers

    # Pass 2: build the real parser now that resources are known.
    parser = argparse.ArgumentParser(parents=[pre])
    parser.add_argument("--target", action="append", required=True,
                        help="resource(s) to build; repeatable")
    resources = sorted(set(_providers.provided) | set(_providers.required))
    for r in resources:
        parser.add_argument(f"--{r.replace('_', '-')}", dest=r, default=None)
    args = parser.parse_args(argv)

    inputs = {r: getattr(args, r) for r in resources if getattr(args, r) is not None}
    result = make(args.target, **inputs)
    # print requested targets
    for t in args.target:
        print(f"{t}:\n{result[t]}\n")
```

**`load_dag_dir`:**

```python
def load_dag_dir(dagdir: Path) -> None:
    # .py providers: pyppin.bulk_import runs the modules, whose @provider
    # decorators register into the global _providers as a side effect.
    bulk_import(str(dagdir), root=str(dagdir))
    # .md stages: factory + register.
    for md in sorted(dagdir.glob("*.md")):
        _providers.add_provider(stage_from_file(md).as_provider())
```

**Decisions / open questions:**
- **Every resource becomes a flag (CONFIRMED).** Matches the existing `make()`
  semantics: supplying a resource skips its provider. Overriding an intermediate
  resource (not just a leaf input) is an intended feature — e.g. pin a
  mid-pipeline value to rerun only the downstream stages. Expose all resources;
  optionally annotate which are leaf inputs in `--help` for readability, but do
  not restrict the set. Flag generation is safe with no sanitization because the
  registration-time invariant guarantees every name is a valid identifier
  (`--<name>` with the standard `-`↔`_` mapping).
- **Flag value types are strings** from argparse. Fine for `.md` template vars
  (stringified anyway) and for `str`-typed function providers. Typed function
  inputs (e.g. `int`, `Path`) would need per-resource `type=`. Plan: look up the
  resource's required type in `_providers` and pass a matching argparse `type=`
  for the simple cases (`int`, `float`, `Path`, `str`); default to `str`.
- **Registry threading:** simplest path uses the module-global `_providers` +
  existing `make()`. If we want test isolation later, add an optional
  `registry: Providers` param to `make()` and an explicit `Providers` in
  `__main__`.
- **LLM backend wiring:** `__main__` should build one shared `ChatBackend`
  (`create_backend()` from item 1, configured via `LLMConfig.add_arguments`/
  `create`) and pass it as `backend=` to every `stage_from_file`, so all stages
  share one provider/config.
- **`--verbose`/`-v` and `--model`** are added explicitly (not as auto-generated
  resource flags), then injected as the reserved `verbose`/`model` resources
  into the `make()` inputs. See reserved-resource convention below.
- `bulk_import` will execute arbitrary `.py` in the dir — document that the dag
  dir is trusted code.

**Suggested build order tomorrow:** Item 1 (backend + Settings) →
Item 2 (+`add_provider`) → Item 3 → Item 5 (file I/O stages) → Item 4 (CLI).
Each step is independently testable with a stub backend.

---

## Cross-cutting: resource names must be valid identifiers

**Invariant (enforce at registration):** every resource name — whether it comes
from a function/parameter name or is passed as a string by a stage factory —
must be a valid Python identifier (`str.isidentifier()`, and reject keywords).
Validate in `Providers.add_provider`/`add` (covering the provider's own name and
every key of `requires`/`optionally_requires`); raise `ValueError` otherwise.

Why: names are meant to be freely mixed between "came from a function" and "just
a string", and to be usable verbatim as an argparse flag, a config-file key,
etc. (Past experience: being able to name anything from a CLI flag or config
file is extremely handy — but only if every name is uniformly safe.) Guaranteeing
identifier-safety up front means Item 4 can generate `--<name>` flags with no
sanitization or reverse-map, and config keys map 1:1 to resource names.

Corollary — **name vs. value.** A name is the resource's DAG identity (an
identifier); its value is the runtime object. A filename like `report.md` is a
*value* (a `Path`), never a name. So a file-writing stage is a resource with an
identifier-safe name (e.g. `report_file`) whose value is the pathname it wrote
(see Item 5).

## Cross-cutting: reserved resource names

Some resource names are **reserved** with framework-wide meaning. They are
ordinary DAG resources (so they flow through `make()` and can be overridden like
any other), but the framework and standard stages agree on their names so behavior
is uniform. Convention: providers/stages declare them as *optional* requirements
(`name | None`) and fall back to a default when absent.

- **`verbose: int`** — uniform debug-verbosity level (0 = quiet; higher = more).
  Any provider may optionally consume it to gate debug output, giving one
  consistent debug API across hand-written providers, LLM stages, and the LLM
  backend. The CLI sets it via repeatable `-v`/`--verbose` (count) and/or
  `--verbose N`; `__main__` injects it as the `verbose` input to `make()`. The
  `llm` backend takes `verbose` through `create_backend`/`complete` and uses it
  for request/response debug logging. Default 0 everywhere.
  - *Decision to confirm tomorrow:* a tiny shared helper (e.g.
    `llm`-independent `debug(level, msg, *, verbose)`), versus each provider
    checking the int itself. Lean toward a one-function helper so formatting is
    uniform.

- **`model: str`** — the LLM model to use, consumed (optionally) by every
  `LLMStage` (see Item 2). One `--model` configures all stages; per-stage
  front-matter overrides it.

When `__main__` auto-generates `--flag`s for every resource (Item 4), `verbose`
and `model` are handled by their explicit, nicer flags instead of generic ones.

---

## Settings / credential management (part of Item 1, needs design)

Port clarity-agent's `settings.Settings` (currently a runtime singleton via
`Settings.current()`) into the `llm` package as the persisted configuration /
credential store — we need a real way to configure and manage LLM API keys for
CLI use, not just read them from env each run.

Open questions to resolve tomorrow (get the CLI setup flow right):

- **Storage:** where do persisted keys live? (e.g. `~/.config/aipipe/settings.toml`
  or an OS keyring.) Precedence: explicit CLI flag > env var > stored settings >
  interactive prompt.
- **Setup flow:** a `dag config`/`aipipe config` subcommand (or `--setup`) that
  prompts for provider + key and writes the store, vs. relying on env vars. What
  happens on first run with no credentials — prompt, or print guidance and exit?
- **Relationship to `LLMConfig`:** keep `LLMConfig` as the resolved, in-memory
  view for a given run; `Settings` is the durable backing store it reads from.
  Decide whether to keep `Settings` a singleton or pass it explicitly (prefer
  explicit, to match the no-global-state direction of the rest of the port).
- **Scope of what's stored:** keys/endpoints/auth-mode per provider, default
  provider, default model/tiers.
- Security: never log keys (respect `verbose`); consider file permissions
  (0600) on any on-disk store.

This is the nuanced piece — worth sketching the flow before writing code.

---

## Item 5 — Standard file I/O stages (read / write)

**Goal:** built-in stage factories for reading files into resources and writing
resources out to files, so pipelines can have real inputs/outputs without
hand-written providers.

**Where:** new module `src/dag/file_stages.py`.

These are **provider-generating factory functions** (like `LLMStage.as_provider`):
they build a `Provider` with explicit `requires`/`provides` and register it via
`Providers.add_provider`. They consume the reserved `output_dir`/`input_dir`
resources (optional `Path`) for resolving relative paths.

**Behavior:**

- **read:** consumes a file path, provides its contents as `str` (text) or
  `bytes` (binary).
- **write:** consumes a content resource (+ optional `output_dir`), writes it to
  a file, and provides the written `Path` as its value (so downstream stages can
  depend on "the file was written"). Content type (`str` vs `bytes`) picks
  text/binary mode at runtime via `isinstance`.

Per the naming invariant above, the first argument of each factory is always the
identifier-safe **resource name**; the filename is a **value** (a literal string
here, or later a path resource), never the name.

**Proposed API (to prototype — exact shape TBD):**

```python
# Write: identifier-safe resource name `report_file`; its value is the Path written.
write_file("report_file", content="report", filename="report.md")
# -> provider "report_file": requires "report" (Any) + optional output_dir (Path),
#    writes str/bytes content to (output_dir / "report.md"), provides Path.

# Read: identifier-safe resource name `raw_text`; value is the file contents.
read_file("raw_text", filename="input.txt")            # text by default
read_file("raw_bytes", filename="input.txt", binary=True)
# -> provider "raw_text": requires optional input_dir (Path),
#    reads (input_dir / "input.txt"), provides str (or bytes).
```

**API questions to play with tomorrow:**

- **Literal filename vs. path-from-resource.** The sketch takes a literal
  `filename=` + an optional dir resource (covers the common case). We may also
  want the path itself to be a resource (computed upstream): e.g.
  `read_file("raw_text", path_resource="src_path")`. Consider supporting both
  forms (literal `filename=` xor `path_resource=`). Either way the *resource
  name* stays identifier-safe.
- **Default filename.** Should `filename=` default to something derived from the
  name (e.g. `report_file` → `report_file`), or always be explicit? Lean
  explicit to avoid surprises.
- **What write provides.** Returning the written `Path` as the value lets other
  stages order after it. Alternative: provide nothing / provide the content
  unchanged. Recommend `Path`.
- **Symmetry / naming.** Settle a consistent signature across `read_file` and
  `write_file` (resource name first; `content=`/`filename=` keyword-only).
- `str` vs `bytes`: pick mode by `isinstance` on write; choose by `binary=` flag
  on read. The provided type is then `str`/`bytes` accordingly.

Reserved resources introduced here: **`output_dir: Path | None`**,
**`input_dir: Path | None`** — add them to the reserved-names list once settled.

---

## Future (not tomorrow) — watcher / trigger mechanism

The immediate deliverable is the CLI runner (Item 4): invoke, build targets,
exit. A later feature is a **watcher** that monitors external state (files
appearing, a queue, a clock, an upstream signal) and triggers `make()` runs when
inputs become available — turning the same provider graph into a reactive
pipeline instead of a one-shot CLI. Out of scope for tomorrow; noting it so the
CLI/`make()` boundary stays clean enough to drive from a long-running process
later (e.g. keep `make()` free of `sys.argv`/`print` coupling — those live in
`__main__`).
