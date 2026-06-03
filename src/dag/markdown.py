"""Parsing for the standard "Markdown with optional YAML front matter" format.

A front-matter block is YAML delimited by ``---`` lines at the very start of the
document (the convention used by Jekyll, Hugo, Obsidian, Pandoc, ...); the rest
is the Markdown body.

    >>> doc = MarkdownDocument.parse("---\\ntitle: Hi\\n---\\nBody text")
    >>> doc.front_matter
    {'title': 'Hi'}
    >>> doc.body
    'Body text'
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class MarkdownDocument:
    """A Markdown document split into its YAML front matter and body."""

    body: str
    front_matter: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, text: str) -> MarkdownDocument:
        """Parse text into front matter + body.

        If the text doesn't begin with a well-formed ``---`` ... ``---`` block,
        the whole text is the body and the front matter is empty.

        Raises:
            ValueError: if a front-matter block is present but isn't a YAML
                mapping (e.g. a bare list or scalar).
            yaml.YAMLError: if the front-matter block is malformed YAML.
        """
        lines = text.split("\n")
        if not lines or lines[0].strip() != "---":
            return cls(body=text)

        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                # Each front-matter line was newline-terminated in the source;
                # add the trailing newline back so YAML block scalars (``|``)
                # keep their final newline.
                block = "\n".join(lines[1:i]) + "\n"
                parsed = yaml.safe_load(block) if block.strip() else {}
                if parsed is None:
                    parsed = {}
                if not isinstance(parsed, dict):
                    raise ValueError(
                        "Front matter must be a YAML mapping (key: value pairs), "
                        f"got {type(parsed).__name__}."
                    )
                body = "\n".join(lines[i + 1 :]).lstrip("\n")
                return cls(
                    body=body, front_matter={str(k): v for k, v in parsed.items()}
                )

        # Opening delimiter with no closing one: not front matter — all body.
        return cls(body=text)

    @classmethod
    def from_file(cls, path: str | Path) -> MarkdownDocument:
        """Parse a Markdown file from disk."""
        return cls.parse(Path(path).read_text(encoding="utf-8"))
