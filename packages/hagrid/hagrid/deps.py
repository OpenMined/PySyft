"""The purpose of these functions is to check the local dependencies of the person running the CLI
tool and ensure that things are properly configured for the cli's full use (depending on the user's
operating system.) When dependencies are missing the CLI tool should offer helpful hints about what
course of action to take to install missing dependencies, even offering to run appropriate
installation commands where applicable."""

# stdlib
import platform
import shutil
import subprocess  # nosec
from typing import Any
from typing import Dict
from typing import Optional


class MissingDependency(Exception):
    pass


def get_environment() -> Dict[str, Any]:
    return {
        "uname": platform.uname(),
        "platform": platform.system().lower(),
        "os_version": platform.release(),
        "python_version": platform.python_version(),
    }


ENVIRONMENT = get_environment()


def is_windows() -> bool:
    if "platform" in ENVIRONMENT and ENVIRONMENT["platform"] == "windows":
        return True
    return False


allowed_hosts = ["docker", "vm", "azure", "aws", "gcp"]
commands = ["docker", "git", "ansible-playbook"]

if is_windows():
    commands.append("wsl")


def check_deps() -> Dict[str, Optional[str]]:
    paths = {}
    for dep in commands:
        paths[dep] = shutil.which(dep)
    return paths


DEPENDENCIES = check_deps()


def docker_info() -> str:
    try:
        cmd = "docker info"
        output = subprocess.check_output(cmd, shell=True)  # nosec
        return str(output.decode("utf-8"))
    except Exception as e:
        print("failed to get docker info", e)
        return str(e)


def wsl_info() -> str:
    try:
        cmd = "wsl --status"
        output = subprocess.check_output(cmd, shell=True)  # nosec
        return str(output.decode("utf-8"))
    except Exception as e:
        print("failed to get wsl info", e)
        return str(e)


def wsl_linux_info() -> str:
    try:
        cmd = "wsl bash -c 'lsb_release -a'"
        output = subprocess.check_output(cmd, shell=True)  # nosec
        return str(output.decode("utf-8"))
    except Exception as e:
        print("failed to get wsl linux info", e)
        return str(e)
