# aipipe

A small DAG-based dependency-resolving pipeline builder.

Declare functions as *providers*: each names the resource it provides (its
return value) and the resources it requires (its parameters). Then ask `make`
to build a target, and it resolves and runs the necessary providers in order.

```python
from pathlib import Path

from dag import make, provider


@provider
def first_output(input: Path) -> str:
    return str(input)


@provider
def second_output(input: Path, first_output: str) -> str:
    return first_output.upper()


make("second_output", input=Path("/foo/bar"))
```

## Development

This project uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev   # create the environment and install dependencies
uv run pytest         # run the tests
uv run ruff check     # lint
```
