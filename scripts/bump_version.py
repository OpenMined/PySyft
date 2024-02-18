# The script mainly help in bumping the version of the package
# during release process.

# stdlib
# It covers mainly two scenarios:
# Bumping from beta to the stable version
# Bumping from stable to the next beta version
import argparse
from typing import Optional

# Command line processing


def handle_command_line(arglist: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(description="Bump the version of the package.")
    parser.add_argument(
        "--bump-to-stable",
        type=int,
        help="Bump the current beta version to a stable version.",
    )


if __name__ == "__main__":
    handle_command_line()
