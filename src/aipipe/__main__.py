"""``python -m aipipe`` — the central aipipe command-line tool."""

import sys

from aipipe.cli import main

if __name__ == "__main__":
    sys.exit(main())
