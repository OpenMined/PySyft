# stdlib
import sys

# checks if we are in a python repl or python -i script.py
is_interactive = bool(getattr(sys, "ps1", sys.flags.interactive))

# checks if we are in jupyter
is_jupyter = True

try:
    get_ipython()  # type: ignore
    # third party
    import notebook
    from packaging import version

    NOTEBOOK_VERSION = version.parse(notebook.__version__.split("+")[0])
    if NOTEBOOK_VERSION < version.parse("6.0.0") and "google.colab" not in sys.modules:
        # google.colab check to fix issue #5315
        raise Exception(
            "Your Jupyter Notebook is too old. Please upgrade to version 6 or higher."
        )
except NameError:
    is_jupyter = False

__all__ = ["is_interactive", "is_jupyter"]
