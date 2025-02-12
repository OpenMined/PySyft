import platform

__all__ = ["OS_NAME", "OS_VERSION", "OS_ARCH", "PYTHON_VERSION"]

PYTHON_VERSION = platform.python_version()
UNAME = platform.uname()
OS_NAME = ""
OS_VERSION = ""
OS_ARCH = UNAME.machine
OS_IS_WSL2 = False

if UNAME.system == "Darwin":
    OS_NAME = "macOS"
    OS_VERSION = platform.mac_ver()[0]

elif UNAME.system == "Linux":
    import distro

    OS_NAME = distro.name()
    OS_VERSION = distro.version(best=True)
    OS_IS_WSL2 = "wsl2" in UNAME.release.lower()

    if OS_IS_WSL2:
        OS_NAME = "WSL2-" + OS_NAME

elif UNAME.system == "Windows":
    OS_NAME = UNAME.system
    OS_VERSION = platform.win32_ver()[0]

else:
    OS_NAME = UNAME.system
    OS_VERSION = UNAME.release
