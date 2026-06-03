from pathlib import Path
from pyppin.os.bulk_import import bulk_import
import argparse

def main() -> None:
    # We do our first round of argument parsing with just "core" args that tell us which DAG
    # to load.
    parser = argparse.ArgumentParser(prog="dag", description="DAG runner")
    parser.add_argument(
        "dirname", required=True, type=str, help="The directory where the DAG can be found"
    )
    core_args, resource_args = parser.parse_known_args()

    dagdir = Path(core_args.dirname)
    if not dagdir.is_dir():
        raise ValueError(f"Couldn't find the directory '{dagdir}'")





main()