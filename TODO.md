# TODO

> Detailed implementation plans for the items below: [PLAN.md](PLAN.md).

## Next session

- [ ] **Port the LLM backend from `clarity-agent`.** Bring in its LLM backend
  logic for **all providers** and refactor it into a portable, reusable library
  in a new top-level `src/llm/` package (not under `dag`). Includes porting
  `Settings` for credential/key management — get the CLI setup flow right.

- [ ] **Add an `LLMStage` class.** Takes a Markdown file containing
  `{{substitutions}}`. The substitution variable names become the stage's
  required resources (typed `Any`), and the stage's output is the resource it
  provides. Consumes the reserved `model` resource for model selection.

- [ ] **Add a factory function for `LLMStage`.** Reads an `.md` file and
  returns the corresponding `LLMStage`.

- [ ] **Add standard file I/O stages.** `read_file`/`write_file` factory
  functions that generate file-reading/writing providers (str/bytes), using
  optional `input_dir`/`output_dir` resources. Exact API still to prototype.

- [ ] **Wire up `__main__`.**
  - Bulk-import a `dagdir`, running the factory function over its `.md` files.
  - Re-run the argument parser with every discovered resource exposed as an
    optional flag, plus a flag for the target(s).
  - Call `make()` with the parsed flags as inputs and the requested targets.

- [ ] **Cross-cutting:** reserved `verbose: int` resource for a uniform debug
  API across all providers and the LLM backend (CLI `-v`/`--verbose`).

_Later (not next session):_ a **watcher** mechanism that monitors external state
and triggers `make()` runs when inputs become available.
