from pathlib import Path
from pyppin.base.import_file import import_file
from pyppin.os.bulk_import import bulk_import
import importlib.machinery
from dag.dag import registry
from dag.llm_stage import stage_from_file


def import_providers(src: Path) -> None:
    """Import all Providers (LLM and Python) that you can find at the given path."""

    def _load_md_file(src: Path) -> bool:
        if src.suffix == ".md" and src.stem.isidentifier():
            registry.add(stage_from_file(src).as_provider())
        return True

    if src.is_dir():
        bulk_import(src, visitor=_load_md_file)
    elif src.suffix == ".md":
        _load_md_file(src)
    elif src.suffix in importlib.machinery.all_suffixes():
        import_file(src)
