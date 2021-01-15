# stdlib
import sys

# checks if we are in a python repl or python -i script.py
is_interactive = bool(getattr(sys, "ps1", sys.flags.interactive))

# checks if we are in jupyter
is_jupyter = True

try:
    get_ipython()  # type: ignore
except NameError:
    is_jupyter = False

__all__ = ["is_interactive", "is_jupyter"]
