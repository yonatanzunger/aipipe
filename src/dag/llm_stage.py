"""Turn a Markdown template with ``{{substitutions}}`` into a DAG stage.

An :class:`LLMStage` wraps a prompt template whose ``{{variable}}`` names become
the stage's required resources and whose LLM completion becomes the resource it
provides. Register it with the DAG via :meth:`LLMStage.register` (or
``registry.add(stage.as_provider())``) and build it with :func:`dag.make`.

Example::

    stage = LLMStage("summary", "Summarize this:\\n\\n{{document}}")
    stage.register()
    make("summary", document="...long text...")  # -> {"document": ..., "summary": "<llm output>"}
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llm import complete

from dag.dag import Provider, registry, resource
from dag.logger import Logger
from dag.markdown import MarkdownDocument

if TYPE_CHECKING:
    from llm.chat import ChatBackend

# Matches ``{{ name }}`` (optional surrounding whitespace); ``name`` must be a
# valid identifier so it can be a resource name.
_VAR = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}")

# Reserved resource: the LLM model that stages use. A single ``--model`` (or
# ``make(model=...)``) drives every stage; an individual stage's own model
# (e.g. from front matter) takes precedence over it.
resource("model", str, help="LLM model name to use for LLM stages")


class LLMStage:
    """A DAG provider whose output is an LLM completion of a rendered template.

    The ``output`` type controls how the completion is surfaced:

    - ``str``: the completion text is the resource's value (and is
      substituted inline wherever a downstream stage references it).
    - ``Path`` (default): the completion is written to ``workdir/<name><extension>`` and
      that path becomes the resource's value, so downstream ``{{var}}`` renders
      the *filename* (which an agentic backend can read on its own).

    Args:
        name: The resource this stage provides (must be a valid identifier).
        template: The prompt text, with ``{{var}}`` placeholders.
        output: ``str`` or ``Path`` — see above.
        extension: File extension used when ``output is Path``.
        model: Optional per-stage model override. Highest precedence; falls back
            to the ``model`` resource, then the backend default.
        system: Optional system prompt for the completion.
        backend: Optional pre-built :class:`~llm.chat.ChatBackend`. When omitted,
            a backend is created per call from the ambient configuration.
    """

    def __init__(
        self,
        name: str,
        template: str,
        *,
        output: type = Path,
        extension: str = ".md",
        model: str | None = None,
        system: str | None = None,
        backend: "ChatBackend | None" = None,
    ) -> None:
        self.name = name
        self.template = template
        # De-duplicated, in order of first appearance.
        self.variables: list[str] = list(dict.fromkeys(_VAR.findall(template)))
        self.output = output
        self.extension = extension
        self.model = model
        self.system = system
        self.backend = backend

    def render(self, values: dict[str, Any]) -> str:
        """Substitute ``{{var}}`` placeholders using *values* (stringified)."""
        return _VAR.sub(lambda m: str(values[m.group(1)]), self.template)

    def __call__(self, logger: Logger, **kwargs: Any) -> str | Path:
        """Render the template from the supplied resources and complete it.

        Returns the completion text, or — when ``output is Path`` — the path to
        a file the completion was written to (under the ``workdir`` resource).
        """
        model = self.model or kwargs.get("model")
        prompt = self.render(kwargs)
        logger.log(1, "Model", model)
        logger.log(1, "Prompt", prompt)
        response = complete(
            prompt, system=self.system, model=model, backend=self.backend
        )
        logger.log(1, "Response", response)
        if self.output is Path:
            return self._write(response, kwargs["workdir"], logger)
        return response

    def _write(self, text: str, workdir: Path, logger: Logger) -> Path:
        """Write the completion to ``workdir/<name><extension>`` and return it.

        The directory is created lazily here, so runs with no file-output stages
        leave no empty directories behind.
        """
        workdir.mkdir(parents=True, exist_ok=True)
        path = workdir / f"{self.name}{self.extension}"
        path.write_text(text, encoding="utf-8")
        logger.log(1, "Wrote", path)
        return path

    def as_provider(self) -> Provider:
        """Build the :class:`~dag.dag.Provider` representing this stage.

        Template variables become required resources (typed ``Any``, since they
        are stringified into the prompt); the reserved ``model`` resource is an
        optional requirement so a globally-supplied ``model`` reaches the stage
        without forcing it to be provided. A file-output stage (``output is
        Path``) additionally requires the ``workdir`` it writes into.
        """
        requires: dict[str, Any] = {v: Any for v in self.variables}
        optionally_requires: dict[str, Any] = {}
        if "model" not in requires:
            optionally_requires["model"] = str | None
        # The ambient logger is injected by make(); declare it (optional) so
        # Provider.__call__ forwards it to __call__.
        if "logger" not in requires:
            optionally_requires["logger"] = Logger
        if self.output is Path and "workdir" not in requires:
            requires["workdir"] = Path
        return Provider(
            name=self.name,
            func=self,
            provides=self.output,
            requires=requires,
            optionally_requires=optionally_requires,
        )

    def register(self, into: Any = None) -> None:
        """Register this stage as a provider (default: the global registry)."""
        (into or registry).add(self.as_provider())


_FILE_OUTPUT_WORDS = {"file", "path"}


def stage_from_file(
    path: str | Path,
    *,
    output: type = Path,
    model: str | None = None,
    backend: "ChatBackend | None" = None,
) -> LLMStage:
    """Read a Markdown file into an :class:`LLMStage`.

    The file is parsed as Markdown with optional YAML front matter (see
    :class:`~dag.markdown.MarkdownDocument`). The resource name defaults to the
    file's stem (e.g. ``summary.md`` → ``summary``); front matter may override
    ``name``, set ``model`` / ``system`` / ``extension``, and choose file output
    via ``output: file`` (or ``path``). Front-matter values take precedence over
    the *output* / *model* arguments. Side-effect-free — does not register.
    """
    path = Path(path)
    doc = MarkdownDocument.from_file(path)
    meta = doc.front_matter
    name = str(meta.get("name") or path.stem)
    if not name.isidentifier():
        raise ValueError(
            f"Resource name {name!r} (from {path}) is not a valid Python "
            f"identifier. Rename the file or set `name:` in its front matter."
        )
    fm_output = meta.get("output")
    if fm_output is not None:
        output = Path if str(fm_output).lower() in _FILE_OUTPUT_WORDS else str
    stage = LLMStage(
        name=name,
        template=doc.body,
        output=output,
        extension=meta.get("extension", ".md"),
        model=meta.get("model") or model,
        system=meta.get("system"),
        backend=backend,
    )
    stage.__doc__ = f"Produced by {path}"
    return stage
