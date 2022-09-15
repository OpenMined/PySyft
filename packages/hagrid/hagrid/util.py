# stdlib
import platform
import subprocess


def os_name() -> str:
    os_name = platform.system()
    if os_name.lower() == "darwin":
        return "macOS"
    else:
        return os_name


def verify_git_installation() -> None:

    try:
        subprocess.call("git", stdout=subprocess.DEVNULL)
    except FileNotFoundError:
        print(
            "Hagrid requires git for the initial setup, Kindly follow the link below\n"
            + " to install git on your System."
        )
        os = os_name()

        git_link = {
            "Windows": "https://git-scm.com/download/win",
            "macOS": "https://git-scm.com/download/mac",
            "Linux": "https://git-scm.com/download/linux",
        }
        print(f"{os} : {git_link[os]}")
        exit(0)


verify_git_installation()
