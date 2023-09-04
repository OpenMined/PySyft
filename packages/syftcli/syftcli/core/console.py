# stdlib
from typing import Any

# third party
from rich import get_console
from rich import print

__all__ = ["console", "info", "warn", "success", "error", "debug", "print"]

console = get_console()


def info(*args: Any, **kwargs: Any) -> None:
    console.print(*args, style="bold cyan", **kwargs)


def debug(*args: Any, **kwargs: Any) -> None:
    console.print(*args, style="grey50", highlight=False, **kwargs)


def warn(*args: Any, **kwargs: Any) -> None:
    console.print(*args, style="bold yellow", **kwargs)


def success(*args: Any, **kwargs: Any) -> None:
    console.print(*args, style="bold green", **kwargs)


def error(*args: Any, **kwargs: Any) -> None:
    console.print(*args, style="bold red", **kwargs)
