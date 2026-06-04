# aipipe

A small DAG-based pipeline builder for wiring together ordinary Python functions
and LLM prompts.

You describe a pipeline as a set of **resources** (named, typed values) and
**providers** (functions that turn some resources into another). You then ask
`make` for the resource you want; aipipe resolves the dependency graph and runs
the providers in the right order. LLM stages are just providers whose body is a
prompt — and by default their output is written to a file, so an LLM can read
upstream results from disk rather than having them stuffed into the prompt.

## Install

This project uses [uv](https://docs.astral.sh/uv/). All the supported LLM
provider SDKs are regular dependencies, so there's nothing optional to choose:

```bash
uv sync               # runtime deps
uv sync --extra dev   # + pytest / ruff for development
```

## Configure a provider

aipipe talks to Anthropic, OpenAI, Azure, Gemini, the Claude Agent SDK, and the
GitHub Copilot SDK. Configure one interactively — the wizard walks you through
the provider/auth choices and makes a live test call to confirm it works:

```bash
aipipe config setup
```

Or set things non-interactively. Credentials are stored in your OS keychain;
preferences live in a small `settings.json`. Standard environment variables
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …) are also honored and take precedence.

```bash
aipipe config set ANTHROPIC_API_KEY sk-ant-...   # store a key
aipipe config show                               # what's configured?
aipipe config unset ANTHROPIC_API_KEY            # clear it
```

(`aipipe …` and `python -m aipipe …` are equivalent.)

## Concepts

- A **resource** is a named value with a type. Names must be valid Python
  identifiers, so they double as command-line flags and config keys.
- A **provider** makes one resource from others. The easiest way to write one is
  to decorate a function: the function name is the resource it provides, and its
  parameters are the resources it requires (a `foo | None` parameter is an
  *optional* dependency).
- `make(target, **inputs)` builds `target`, running only the providers needed,
  and returns a dict of every resource produced along the way.

```python
from dag import make, provider


@provider
def greeting(name: str) -> str:
    return f"Hello, {name}!"


@provider
def shout(greeting: str) -> str:
    return greeting.upper()


make("shout", name="World")
# {'name': 'World', 'greeting': 'Hello, World!', 'shout': 'HELLO, WORLD!'}
```

## LLM stages from Markdown

An LLM stage is a Markdown file whose `{{placeholders}}` become its required
resources and whose completion becomes the resource it provides. The file's name
is the resource name; optional YAML front matter sets the model, system prompt,
and output type.

`pipeline/outline.md`:

```markdown
---
system: You are a sharp technical editor.
model: claude-sonnet-4-6
---
Write a tight outline for an article about {{topic}}.
```

`pipeline/article.md`:

```markdown
Using the outline in {{outline}}, write the full article.
```

**Output is a file by default.** Each stage writes its result to
`.aipipe/runs/<timestamp>/<stage>.md` and the resource's value is that `Path`.
So `{{outline}}` above is substituted with the *path* to `outline.md`, and an
agentic backend (the Claude/Copilot SDKs) reads it on its own. Add
`output: str` to a stage's front matter if you'd rather its value be the
completion text inlined directly into downstream prompts.

## Running a pipeline

Point the CLI at your stage files (and/or `.py` files defining `@provider`
functions) with `--import`, name the target(s) to build, and pass any leaf
inputs as flags — the available flags are generated from whatever you loaded:

```bash
# Build `article`; aipipe runs outline -> article, supplying --topic.
aipipe --import pipeline make article --topic "tidal energy"

# -v / --verbosity shows the plan, prompts, and where files were written.
aipipe -i pipeline make article --topic "tidal energy" -v 1

# Override the model for this run, or pin a per-stage level with --vmodule.
aipipe -i pipeline make article --topic "tidal energy" --model claude-opus-4-7
```

Results land in `.aipipe/runs/<timestamp>/` (override with `--workdir DIR`); the
directory is created only when a file-output stage actually writes, so dry/string
runs leave nothing behind.

You can also drive everything from Python:

```python
from dag import make
from dag.loader import import_providers

import_providers("pipeline")
result = make("article", topic="tidal energy")
print(result["article"])  # a Path to the written file
```

### Reserved resources

A few resources have framework-wide meaning and can be set on the command line:

- **`model`** (`--model`) — the LLM model for every stage; a stage's front
  matter overrides it.
- **`verbosity`** (`-v` / `--verbosity`) — debug-output level (`0` = quiet).
- **`vmodule`** (`--vmodule`) — per-stage verbosity, e.g. `outline:2,article:1`.
- **`workdir`** (`--workdir`) — where file-output stages write (defaults to a
  fresh per-run directory).

## Development

```bash
uv run pytest         # run the tests
uv run ruff check     # lint
```
