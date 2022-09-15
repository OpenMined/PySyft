# stdlib
import platform


def os_name() -> str:
    os_name = platform.system()
    if os_name.lower() == "darwin":
        return "macOS"
    else:
        return os_name
