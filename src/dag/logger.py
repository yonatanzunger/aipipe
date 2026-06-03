import os
import re
import sys
import textwrap
from typing import Any, TextIO

from pyppin.base.lazyinit import lazyinit
from pyppin.text.tty import TTY, tty


class Logger(object):
    def __init__(
        self, verbosity: int, outfile: TextIO = sys.stdout, textwidth: int | None = None
    ):
        self.verbosity = verbosity
        self.outfile = outfile
        self._textwidth = textwidth

    @lazyinit
    def textwidth(self) -> int:
        if self._textwidth is not None:
            return self._textwidth
        try:
            dims = os.get_terminal_size(self.outfile.fileno())
            return dims.columns
        except OSError:
            return 100  # Default

    def log(
        self,
        level: int,
        title: str,
        text: Any,
        indent: int = 0,
        textwidth: int | None = None,
        truncate: bool = True,
    ) -> None:
        """Print something out.

        Args:
            level: The min verbosity we need to be at in order to print this out. 0 means unconditional print,
                1, 2, 3... reflect higher and higher levels of debug.
            title: A title for what you're outputting, e.g. "Target" or "Input".
            text: The thing you actually want to print out. We'll call str() on it.
            textwidth: The console width at which to print (if you want to override the default behavior)
            indent: The number of initial spaces of indent to add.
            truncate: If true and our current verbosity level <= the target log level, truncate to fit on screen.
        """
        if level > self.verbosity:
            return

        # We hold off calling str() until now because it may be expensive and we often fast-exit above.
        text = str(text)
        indent_str = " " * indent
        header = (
            indent_str + tty(TTY.BRIGHT, text=f"{title}: ", file=self.outfile)
            if title
            else indent_str
        )
        textwidth = textwidth if textwidth is not None else self.textwidth

        if len(header) + len(text) < textwidth:
            self.outfile.write(f"{header}{text}\n")
        elif truncate and level == self.verbosity:
            self.outfile.write(f"{header}{text[: textwidth - len(header) - 3]}...\n")
        else:
            self.outfile.write(header + "\n")
            subindent = indent_str + "  "
            for line in textwrap.wrap(
                text,
                textwidth,
                initial_indent=subindent,
                subsequent_indent=subindent,
            ):
                self.outfile.write(line + "\n")

    def header(self, level: int, title: str, indent: int = 0) -> None:
        """Print a bright title-only line (e.g. a section header)."""
        if level > self.verbosity:
            return
        prefix = " " * indent
        self.outfile.write(prefix + tty(TTY.BRIGHT, text=f"{title}:", file=self.outfile) + "\n")


class LoggerFactory(object):
    """Parse verbosity/vmodule settings and build per-stage :class:`Logger`s.

    ``verbosity`` is the global level; ``vmodule`` (``"stage:level,..."``)
    overrides it per provider. :meth:`logger` produces a :class:`Logger` at the
    right level for a given stage (or the global level when no stage is given).
    """

    PATTERN = re.compile("([a-zA-Z_][a-zA-Z0-9_]*):([0-9]+)")

    def __init__(self, resources: dict[str, Any]):
        self.verbosity: int = resources.get("verbosity") or 0
        self.vmodule: dict[str, int] = {}
        if "vmodule" in resources:
            for entry in resources["vmodule"].split(","):
                if not entry:
                    continue
                match = self.PATTERN.fullmatch(entry)
                if match is None:
                    raise ValueError(
                        f"Unexpected value in --vmodule '{entry}': "
                        "should look like 'resource:level'"
                    )
                self.vmodule[match[1]] = int(match[2])

    def level(self, stage: str | None = None) -> int:
        """The verbosity level for *stage* (falls back to the global level)."""
        if stage is None:
            return self.verbosity
        return self.vmodule.get(stage, self.verbosity)

    def logger(self, stage: str | None = None) -> Logger:
        """A :class:`Logger` at the level appropriate for *stage*."""
        return Logger(self.level(stage))
