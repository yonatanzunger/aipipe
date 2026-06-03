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

    Args:
        name: The resource this stage provides (must be a valid identifier).
        template: The prompt text, with ``{{var}}`` placeholders.
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
        model: str | None = None,
        system: str | None = None,
        backend: "ChatBackend | None" = None,
    ) -> None:
        self.name = name
        self.template = template
        # De-duplicated, in order of first appearance.
        self.variables: list[str] = list(dict.fromkeys(_VAR.findall(template)))
        self.model = model
        self.system = system
        self.backend = backend

    def render(self, values: dict[str, Any]) -> str:
        """Substitute ``{{var}}`` placeholders using *values* (stringified)."""
        return _VAR.sub(lambda m: str(values[m.group(1)]), self.template)

    def __call__(self, logger: Logger, **kwargs: Any) -> str:
        """Render the template from the supplied resources and complete it."""
        model = self.model or kwargs.get("model")
        prompt = self.render(kwargs)
        logger.log(1, "Model", model)
        logger.log(1, "Prompt", prompt)
        response = complete(
            prompt, system=self.system, model=model, backend=self.backend
        )
        logger.log(1, "Response", response)
        return response

    def as_provider(self) -> Provider:
        """Build the :class:`~dag.dag.Provider` representing this stage.

        Template variables become required resources (typed ``Any``, since they
        are stringified into the prompt); the reserved ``model`` resource is an
        optional requirement so a globally-supplied ``model`` reaches the stage
        without forcing it to be provided.
        """
        requires: dict[str, Any] = {v: Any for v in self.variables}
        optionally_requires: dict[str, Any] = {}
        if "model" not in requires:
            optionally_requires["model"] = str | None
        # The ambient logger is injected by make(); declare it (optional) so
        # Provider.__call__ forwards it to __call__.
        if "logger" not in requires:
            optionally_requires["logger"] = Logger
        return Provider(
            name=self.name,
            func=self,
            provides=str,
            requires=requires,
            optionally_requires=optionally_requires,
        )

    def register(self, into: Any = None) -> None:
        """Register this stage as a provider (default: the global registry)."""
        (into or registry).add(self.as_provider())


def stage_from_file(
    path: str | Path,
    *,
    model: str | None = None,
    backend: "ChatBackend | None" = None,
) -> LLMStage:
    """Read a Markdown file into an :class:`LLMStage`.

    The file is parsed as Markdown with optional YAML front matter (see
    :class:`~dag.markdown.MarkdownDocument`). The resource name defaults to the
    file's stem (e.g. ``summary.md`` → ``summary``); front matter may override
    ``name`` and set ``model`` / ``system``. Front-matter ``model`` takes
    precedence over the *model* argument. Side-effect-free — does not register.
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
    return LLMStage(
        name=name,
        template=doc.body,
        model=meta.get("model") or model,
        system=meta.get("system"),
        backend=backend,
    )
