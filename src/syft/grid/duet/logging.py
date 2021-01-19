from typing import Any


def try_print(*args: Any, **kargs: Any) -> None:
    try:
        print(*args, **kargs)
    except BaseException as e:
        print("failed to print msg", e)
