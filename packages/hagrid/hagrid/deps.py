"""The purpose of these functions is to check the local dependencies of the person running the CLI
tool and ensure that things are properly configured for the cli's full use (depending on the user's
operating system.) When dependencies are missing the CLI tool should offer helpful hints about what
course of action to take to install missing dependencies, even offering to run appropriate
installation commands where applicable."""

# future
from __future__ import annotations

# stdlib
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
import getpass
import json
import os
import platform
import re
import shutil
import subprocess  # nosec
import sys
import traceback
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# third party
from packaging import version
from packaging.version import Version
import requests
from rich.console import Console

# relative
from .exceptions import MissingDependency
from .lib import is_gitpod
from .mode import EDITABLE_MODE
from .nb_output import NBOutput
from .version import __version__

LATEST_STABLE_SYFT = "0.6"

DOCKER_ERROR = """
You are running an old version of docker, possibly on Linux. You need to install v2.
At the time of writing this, if you are on linux you need to run the following:

DOCKER_COMPOSE_VERSION=v2.7.0
curl -sSL https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-linux-x86_64 \
     -o ~/.docker/cli-plugins/docker-compose
chmod +x ~/.docker/cli-plugins/docker-compose

ALERT: you may need to run the following command to make sure you can run without sudo.

echo $USER              //(should return your username)
sudo usermod -aG docker $USER

... now LOG ALL THE WAY OUT!!!

...and then you should be good to go. You can check your installation by running:

docker compose version
"""

SYFT_MINIMUM_PYTHON_VERSION = (3, 7)
SYFT_MINIMUM_PYTHON_VERSION_STRING = "3.7"
SYFT_MAXIMUM_PYTHON_VERSION = (3, 10, 999)
SYFT_MAXIMUM_PYTHON_VERSION_STRING = "3.10"
WHITE = "\033[0;37m"
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
BOLD = "\033[1m"
NO_COLOR = "\033[0;0m"
WARNING_MSG = f"\033[0;33mWARNING:{NO_COLOR}"


def get_version_string() -> str:
    version = str(__version__)
    if EDITABLE_MODE:
        version += "-dev"
    return version


@dataclass
class SetupIssue:
    issue_name: str
    description: str
    command: Optional[str] = None
    solution: Optional[str] = None


@dataclass
class Dependency:
    of: str = ""
    name: str = ""
    display: str = ""
    only_os: str = ""
    version: Optional[Version] = version.parse("0.0")
    valid: bool = False
    issues: List[SetupIssue] = field(default_factory=list)
    output_in_text: bool = False

    def check(self) -> None:
        pass


@dataclass
class DependencySyftOS(Dependency):
    of: str = "syft"

    def check(self) -> None:
        self.display = "✅ " + ENVIRONMENT["os"]
        if is_windows():
            if not get_pip_package("jaxlib"):
                self.issues.append(windows_jaxlib())
        elif is_apple_silicon():
            pass


@dataclass
class DependencySyftPython(Dependency):
    of: str = "syft"

    def check(self) -> None:
        self.version = sys.version_info
        if (
            sys.version_info >= SYFT_MINIMUM_PYTHON_VERSION
            and sys.version_info <= SYFT_MAXIMUM_PYTHON_VERSION
        ):
            self.display = "✅ Python " + ENVIRONMENT["python_version"]
        else:
            self.issues.append(python_version_unsupported())
            self.display = "❌ " + ENVIRONMENT["python_version"]


@dataclass
class DependencyGridGit(Dependency):
    of: str = "grid"

    def check(self) -> None:
        binary_info = BinaryInfo(
            binary="git", version_cmd="git --version"
        ).get_binary_info()
        if binary_info.path and binary_info.version:
            self.display = "✅ Git " + str(binary_info.version)
        else:
            self.issues.append(git_install(self.output_in_text))
            self.display = "❌ Git not installed"


MINIMUM_DOCKER_VERSION = "20.0.0"


@dataclass
class DependencyGridDocker(Dependency):
    of: str = "grid"

    def check(self) -> None:
        binary_info = BinaryInfo(
            binary="docker", version_cmd="docker --version"
        ).get_binary_info()
        if binary_info.path and binary_info.version > version.parse(
            MINIMUM_DOCKER_VERSION
        ):
            self.display = "✅ Docker " + str(binary_info.version)
        else:
            self.issues.append(docker_install())
            self.display = "❌ Docker not installed"


MINIMUM_DOCKER_COMPOSE_VERSION = "2.0.0"


@dataclass
class DependencyGridDockerCompose(Dependency):
    of: str = "grid"

    def check(self) -> None:
        binary_info = BinaryInfo(
            binary="docker", version_cmd="docker compose version"
        ).get_binary_info()

        if (
            binary_info.path
            and binary_info.version
            and binary_info.version > version.parse(MINIMUM_DOCKER_COMPOSE_VERSION)
        ):
            self.display = "✅ Docker Compose " + str(binary_info.version)
        else:
            self.issues.append(docker_compose_install())
            self.display = "❌ Docker Compose v2 not installed"


@dataclass
class DependencyPyPI(Dependency):
    of: str = "none"
    package_name: str = ""
    package_display_name: str = ""
    pre: bool = False
    install_issue: Callable = lambda: None
    update_available_issue: Callable = lambda: None

    def check(self) -> None:
        package_dict = get_pip_package(self.package_name)

        if package_dict is None:
            self.display = "❌ " + f"{self.package_display_name} not installed"
            self.issues.append(self.install_issue(pre=self.pre))
        else:
            version_string = package_dict["version"]
            current_version = version.parse(version_string)
            if "editable_project_location" in package_dict:
                self.display = (
                    "🚨 "
                    + f"{self.package_name}=={str(current_version)} -e {package_dict['editable_project_location']}"
                )
            else:
                is_newer, latest_version = new_pypi_version(
                    package=self.package_name, current=current_version, pre=self.pre
                )
                if not is_newer:
                    channel = "stable"
                    if current_version.is_prerelease:
                        channel = "pre-release"
                    self.display = (
                        "✅ "
                        + f"{self.package_name}=={str(version_string)} (latest {channel})"
                    )
                else:
                    self.display = (
                        "✅ "
                        + f"{self.package_name}=={str(current_version)} (Version {str(latest_version)} available)"
                    )
                    self.issues.append(
                        self.update_available_issue(current_version, latest_version)
                    )


def new_pypi_version(
    package: str, current: Version, pre: bool = False
) -> Tuple[bool, Version]:
    pypi_json = get_pypi_versions(package_name=package)
    if (
        "info" not in pypi_json
        or "releases" not in pypi_json
        or "version" not in pypi_json["info"]
    ):
        raise Exception("Bad response from PyPi")

    if not current.is_prerelease and not pre:
        latest_stable = version.parse(pypi_json["info"]["version"])
        if current < latest_stable:
            return (True, latest_stable)
        else:
            return (False, current)
    else:
        latest_release = current

        releases = sorted(list(pypi_json["releases"].keys()))
        for release in releases:
            pre_release_version = version.parse(release)
            if latest_release < pre_release_version:
                latest_release = pre_release_version

        if latest_release != current:
            return (True, latest_release)
        else:
            return (False, latest_release)


def get_pypi_versions(package_name: str) -> Dict[str, Any]:
    try:
        pypi_url = f"https://pypi.org/pypi/{package_name}/json"
        req = requests.get(pypi_url)
        # TODO: Fix JSON parsing of version keys
        # this is broken on my machine for some reason, the version keys are wrong
        pypi_info = json.loads(req.text)
        # print(pypi_info["releases"].keys())
        return pypi_info

    except Exception as e:
        print(f"Unable to get JSON from PyPI URL: {pypi_url}. {e}")
        raise e


def get_pip_package(package_name: str) -> Optional[Dict[str, str]]:
    packages = get_pip_packages()
    for package in packages:
        if package["name"] == package_name:
            return package
    return None


def get_pip_packages() -> List[Dict[str, str]]:
    try:
        cmd = "python -m pip list --format=json --disable-pip-version-check"
        output = subprocess.check_output(cmd, shell=True)  # nosec
        return json.loads(str(output.decode("utf-8")).strip())
    except Exception as e:
        print("failed to pip list", e)
        raise e


def get_location(binary: str) -> Optional[str]:
    return shutil.which(binary)


@dataclass
class BinaryInfo:
    binary: str
    version_cmd: str
    error: Optional[str] = None
    path: Optional[str] = None
    version: Optional[Union[str, Version]] = version.parse("0.0")
    version_regex = (
        r"[^\d]*("
        + r"(0|[1-9][0-9]*)\.*(0|[1-9][0-9]*)\.*(0|[1-9][0-9]*)"
        + r"(-((0|[1-9][0-9]*|[0-9]*[a-zA-Z-][0-9a-zA-Z-]*)"
        + r"(\.(0|[1-9][0-9]*|[0-9]*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
        + r"(\+([0-9a-zA-Z-]+(\.[0-9a-zA-Z-]+)*))?)"
        + r"[^\d].*"
    )

    def extract_version(self, lines: List[str]) -> None:
        for line in lines:
            matches = re.match(self.version_regex, line)
            if matches is not None:
                self.version = matches.group(1)
                try:
                    if "-gitpod" in self.version:
                        parts = self.version.split("-gitpod")
                        self.version = parts[0]
                    self.version = version.parse(self.version)
                except Exception:  # nosec
                    pass
                break

    def get_binary_info(self) -> BinaryInfo:
        self.path = get_location(self.binary)
        if self.path:
            returncode, lines = get_cli_output(self.version_cmd)
            if returncode == 0:
                self.extract_version(lines=lines)
            else:
                if len(lines) > 0:
                    self.error = lines[0]
                else:
                    self.error = f"Error, no output from {self.binary}"
        return self


def get_cli_output(cmd: str, timeout: Optional[float] = None) -> Tuple[int, List[str]]:
    try:
        proc = subprocess.Popen(  # nosec
            cmd.split(" "),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        lines = []
        if proc.stdout and hasattr(proc.stdout, "readlines"):
            lines = [line.decode("utf-8") for line in proc.stdout.readlines()]

        if proc.stderr and hasattr(proc.stderr, "readlines"):
            lines.extend([line.decode("utf-8") for line in proc.stderr.readlines()])

        proc.communicate(timeout=timeout)
        return (int(proc.returncode), lines)
    except Exception as e:
        return (-1, [str(e)])


def gather_debug() -> Dict[str, Any]:
    # relative
    from .lib import commit_hash
    from .lib import hagrid_root

    now = datetime.now().astimezone()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S %Z")
    debug_info: Dict[str, Any] = {}
    debug_info["datetime"] = dt_string
    debug_info["python_binary"] = sys.executable
    debug_info["dependencies"] = DEPENDENCIES
    debug_info["environment"] = ENVIRONMENT
    debug_info["hagrid"] = get_version_string()
    debug_info["hagrid_dev"] = EDITABLE_MODE
    debug_info["hagrid_path"] = hagrid_root()
    debug_info["hagrid_repo_sha"] = commit_hash()
    debug_info["docker"] = docker_info()
    if is_windows():
        debug_info["wsl"] = wsl_info()
        debug_info["wsl_linux"] = wsl_linux_info()
    return debug_info


def get_environment() -> Dict[str, Any]:
    return {
        "uname": platform.uname(),
        "platform": platform.system().lower(),
        "os_version": platform.release(),
        "python_version": platform.python_version(),
    }


ENVIRONMENT = get_environment()


def os_name() -> str:
    os_name = platform.system()
    if os_name.lower() == "darwin":
        return "macOS"
    else:
        return os_name


ENVIRONMENT["os"] = os_name()


def is_apple_silicon() -> bool:
    if (
        "platform" in ENVIRONMENT
        and ENVIRONMENT["platform"].lower() == "darwin"
        and ENVIRONMENT["uname"].machine != "x86_64"
    ):
        return True
    return False


ENVIRONMENT["apple_silicon"] = is_apple_silicon()


def is_windows() -> bool:
    if "platform" in ENVIRONMENT and ENVIRONMENT["platform"].lower() == "windows":
        return True
    return False


allowed_hosts = ["docker", "azure", "aws", "gcp"]
commands = ["docker", "git", "ansible-playbook"]

if is_windows():
    commands.append("wsl")


def check_deps_old() -> Dict[str, Optional[str]]:
    paths = {}
    for dep in commands:
        paths[dep] = shutil.which(dep)
    return paths


DEPENDENCIES = check_deps_old()


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


def check_docker_version() -> Optional[str]:
    if is_windows():
        return "N/A"  # todo fix to work with windows
    result = os.popen("docker compose version", "r").read()  # nosec
    version = None
    if "version" in result:
        version = result.split()[-1]
    else:
        print("This may be a linux machine, either that or docker compose isn't s")
        print("Result:" + result)
        out = subprocess.run(  # nosec
            ["docker", "compose"], capture_output=True, text=True
        )
        if "'compose' is not a docker command" in out.stderr:
            raise MissingDependency(DOCKER_ERROR)

    return version


def docker_running(timeout: Optional[float] = None) -> Tuple[bool, str]:

    status, error_msg = False, ""

    try:
        cmd = "docker info"
        returncode, msg = get_cli_output(cmd, timeout=timeout)
        if returncode == 0:
            status, error_msg = True, "✅ Docker service is running"
        else:
            error_msg = f"""❌ Docker service is either not installed or running.\n\n
To install docker, execute the following steps:\n
1 - Install docker on your machine by using the proper steps according to your OS.\n
{WHITE}MacOS: {GREEN}brew install --cask docker
{WHITE}Linux: {GREEN}curl -fsSL https://get.docker.com -o get-docker.sh && chmod +777 get-docker.sh && ./get-docker.sh
{WHITE}Windows: {GREEN}choco install docker-desktop -y{NO_COLOR} \n
2 - Run \'{GREEN}sudo usermod -a -G docker $USER\'{WHITE} to enable this user to execute docker.
3 - log out and log back in so that your group membership is re-evaluated {NO_COLOR}.
-------------------------------------------------------------------------------------------------------\n
To start your docker service:\n
1 - {WHITE}MacOS/Windows: One can start docker by clicking on the "Docker" icon in your Applications folder.{NO_COLOR}
2 - {WHITE}Ubuntu: {GREEN}sudo service docker start {NO_COLOR}
-------------------------------------------------------------------------------------------------------\n
"""
        error_msg += f"""{YELLOW}{BOLD}Std Output Logs{NO_COLOR}
=================\n\n""" + "\n".join(
            msg
        )

    except Exception as e:  # nosec
        error_msg = str(e)

    return status, error_msg


def allowed_to_run_docker() -> Tuple[bool, str]:
    bool_result, msg = True, ""
    if platform.system().lower() == "linux":
        _, line = get_cli_output("getent group docker")

        # get user
        user = getpass.getuser()

        # Check if current user is root.
        if os.geteuid() == 0:
            bool_result = True

        # Check if current user is member of docker group.
        elif not is_gitpod() and user not in "".join(line):
            msg = f"""⚠️  User is not a member of docker group.
{WHITE}You're currently not allowed to run docker, perform the following steps:\n
    1 - Run \'{GREEN}sudo usermod -a -G docker $USER\'{WHITE} to add docker permissions.
    2 - log out and log back in so that your group membership is re-evaluated {NO_COLOR}."""
            # NOTE: For some reason, inside of CI pipeline the user (runner) isn't a member of
            # docker group and doesn't have sudo priviledges, but can execute docker without
            # permission issues. This is just a workaround to avoid raising an exeception
            # in this scenario without reason.
            if user == "runner":
                bool_result = True
            else:
                bool_result = False

    return bool_result, msg


def check_docker_service_status(animated: bool = True) -> None:
    """Check the status of the docker service.

    Raises:
        MissingDependency: If docker service is not running.
    """

    if not animated:
        docker_installed, msg = docker_running(timeout=60)
        user_allowed, permission_msg = allowed_to_run_docker()
    else:
        console = Console()
        # putting \t at the end seems to prevent weird chars getting outputted
        # during animations in the juypter notebook
        with console.status("[bold blue]Checking for Docker Service[/bold blue]\t"):
            docker_installed, msg = docker_running(timeout=60)
            user_allowed, permission_msg = allowed_to_run_docker()

    # Check if user is allowed to execute docker
    if not user_allowed:
        raise MissingDependency(permission_msg)

    # If docker bin was not found.
    if not docker_installed:
        raise MissingDependency(msg)

    print("✅ Docker service is running")


def check_deps(
    deps: Dict[str, Dependency],
    of: str = "",
    display: bool = True,
    output_in_text: bool = False,
) -> Union[Dict[str, Dependency], NBOutput]:
    output = ""
    if len(of) > 0:
        of = f" {of}"
    # output += f"Checking{of} Dependencies:\n"
    issues = []
    for name, dep in deps.items():
        dep.check()
        output += (dep.display + "\n") if display else ""
        issues += dep.issues

    if not output_in_text:
        if len(issues) > 0:
            output += "<h4>🚨 Some issues were found</h4>"
            for issue in issues:
                output += f"<h5><strong>Issue</strong>: {issue.description}</h5>"
                if issue.solution != "":
                    output += f"<strong>Solution</strong>:\n{issue.solution}"
                if issue.command != "":
                    output += (
                        "<blockquote><strong>Command</strong>:\n "
                        + f"<tt>[ ]</tt><code>!{issue.command}</code></blockquote>"
                    )
                output += "\n"

        return NBOutput(output).to_html()
    else:
        if len(issues) > 0:
            output += "🚨 Some issues were found\n"
            for issue in issues:
                output += f"\nIssue: {issue.description}\n"
                if issue.solution != "":
                    output += f"\nSolution:\n{issue.solution}\n"
                if issue.command != "":
                    output += "\nCommand:\n" + f"{issue.command} "
                output += "\n"

        if len(output) > 0:
            print(output)
        return None  # type: ignore


def check_grid_docker(
    display: bool = True, output_in_text: bool = False
) -> Union[Dict[str, Dependency], NBOutput]:
    try:
        deps: Dict[str, Dependency] = {}
        deps["git"] = DependencyGridGit(name="git")
        deps["docker"] = DependencyGridDocker(name="docker")
        deps["docker_compose"] = DependencyGridDockerCompose(name="docker compose")
        return check_deps(
            of="Grid", deps=deps, display=display, output_in_text=output_in_text
        )
    except Exception as e:
        try:
            if display and not output_in_text:
                return NBOutput(debug_exception(e=e)).to_html()
        except Exception:  # nosec
            pass

        print(e)
        raise e


def debug_exception(e: Exception) -> str:
    exception = (
        f'<div class="alert-danger">An exception occured: {e}.<br />'
        + "Please file a bug report on GitHub Issues or in Slack #support</div>"
    )
    exception += "\n"
    exception += ".\n"
    exception += "https://slack.openmined.org/\n"
    exception += "https://github.com/OpenMined/PySyft/issues\n"
    exception += "\n\nWhen reporting bugs, please copy everything between the lines.\n"
    exception += "==================================================================\n"
    exception += (
        "<code>" + json.dumps(gather_debug(), indent=4, sort_keys=True) + "</code>"
    )
    exception += "\n"
    exception += traceback.format_exc()
    exception += (
        "\n=================================================================\n\n"
    )
    return exception


def check_syft_deps(display: bool = True) -> Union[Dict[str, Dependency], NBOutput]:
    try:
        deps: Dict[str, Dependency] = {}
        deps["os"] = DependencySyftOS(name="os")
        deps["python"] = DependencySyftPython(name="python")
        return check_deps(of="Syft", deps=deps, display=display)
    except Exception as e:
        try:
            if display:
                return NBOutput(debug_exception(e=e)).to_html()
        except Exception:  # nosec
            pass

        print(e)
        raise e


def check_hagrid(display: bool = True) -> Union[Dict[str, Dependency], NBOutput]:
    try:
        deps: Dict[str, Dependency] = {}
        deps["hagrid"] = DependencyPyPI(
            package_name="hagrid",
            package_display_name="HAGrid",
            update_available_issue=hagrid_update_available,
        )
        return check_deps(deps=deps, display=display)
    except Exception as e:
        try:
            if display:
                return NBOutput(debug_exception(e=e)).to_html()
        except Exception:  # nosec
            pass

        print(e)
        raise e


def check_syft(
    display: bool = True, pre: bool = False
) -> Union[Dict[str, Dependency], NBOutput]:
    try:
        deps: Dict[str, Dependency] = {}
        deps["os"] = DependencySyftOS(name="os")
        deps["python"] = DependencySyftPython(name="python")
        deps["syft"] = DependencyPyPI(
            package_name="syft",
            package_display_name="Syft",
            pre=pre,
            install_issue=syft_install,
            update_available_issue=syft_update_available,
        )
        return check_deps(deps=deps, display=display)
    except Exception as e:
        try:
            if display:
                return NBOutput(debug_exception(e=e)).to_html()
        except Exception:  # nosec
            pass

        print(e)
        raise e


PACKAGE_MANAGER_COMMANDS = {
    "git": {
        "macos": "brew install git",
        "windows": 'choco install git.install --params "/GitAndUnixToolsOnPath /WindowsTerminal /NoAutoCrlf" -y',
        "linux": "sudo apt update && sudo apt install git",
        "backup_url": "https://git-scm.com/downloads",
    },
    "docker": {
        "macos": "brew install --cask docker",
        "windows": "choco install docker-desktop -y",
        "linux": "curl -fsSL https://get.docker.com -o get-docker.sh && chmod +777 get-docker.sh && ./get-docker.sh",
        "backup_url": "https://www.docker.com/products/docker-desktop/",
    },
    "docker_compose": {
        "macos": "brew install --cask docker",
        "windows": "choco install docker-desktop -y",
        "linux": (
            "mkdir -p ~/.docker/cli-plugins\n"
            + "DOCKER_COMPOSE_VERSION=v2.7.0\n"
            + "curl -sSL https://github.com/docker/compose/releases/download/"
            + "${DOCKER_COMPOSE_VERSION}/docker-compose-linux-x86_64 "
            + "-o ~/.docker/cli-plugins/docker-compose\n"
            + "chmod +x ~/.docker/cli-plugins/docker-compose"
        ),
        "backup_url": "https://github.com/docker/compose",
    },
}

PACKAGE_MANAGERS = {
    "macos": "brew",
    "windows": "choco",
    "linux": "apt",
}


def os_package_manager_install_cmd(
    package_name: str, package_display_name: str, output_in_text: bool = False
) -> Tuple[Optional[str], Optional[str]]:
    os = ENVIRONMENT["os"].lower()
    cmd = None
    url = None
    package_manager = PACKAGE_MANAGERS[os]
    if (
        package_name in PACKAGE_MANAGER_COMMANDS
        and os in PACKAGE_MANAGER_COMMANDS[package_name]
    ):
        cmd = PACKAGE_MANAGER_COMMANDS[package_name][os]
    if (
        package_name in PACKAGE_MANAGER_COMMANDS
        and "backup_url" in PACKAGE_MANAGER_COMMANDS[package_name]
    ):
        url = PACKAGE_MANAGER_COMMANDS[package_name]["backup_url"]

    solution = ""

    if not output_in_text:
        if cmd:
            solution += f"- You can install {package_display_name} with <code>{package_manager}</code>\n"
        if url:
            if cmd:
                solution += "- Alternatively, you "
            else:
                solution += "- You "
            solution += f"can download and install {package_display_name}"
            solution += f'from <a href="{url}" target="_blank">{url}</a>'
    else:
        if cmd:
            solution += (
                f"- You can install {package_display_name} with {package_manager}\n"
            )
        if url:
            if cmd:
                solution += "- Alternatively, you "
            else:
                solution += "- You "
            solution += f"can download and install {package_display_name} from {url}"

    return (cmd, solution)


def docker_compose_install() -> SetupIssue:
    command, solution = os_package_manager_install_cmd(
        package_name="docker_compose", package_display_name="Docker Compose"
    )
    return SetupIssue(
        issue_name="docker_compose_install",
        description="You do not have Docker Compose v2 installed.",
        command=command,
        solution=solution,
    )


def docker_install() -> SetupIssue:
    command, solution = os_package_manager_install_cmd(
        package_name="docker", package_display_name="Docker"
    )
    return SetupIssue(
        issue_name="docker_install",
        description="You do not have Docker installed.",
        command=command,
        solution=solution,
    )


def git_install(output_in_text: bool = False) -> SetupIssue:
    command, solution = os_package_manager_install_cmd(
        package_name="git", package_display_name="Git", output_in_text=output_in_text
    )
    return SetupIssue(
        issue_name="git_install",
        description="You do not have Git installed.",
        command=command,
        solution=solution,
    )


def syft_install(pre: bool = False) -> SetupIssue:
    command = "pip install -U syft --pre"
    if pre:
        # command += " --pre"
        pass
    return SetupIssue(
        issue_name="syft_install",
        description="You have not installed Syft.",
        command=command,
        solution="You can install Syft with pip.",
    )


def syft_update_available(current_version: Version, new_version: Version) -> SetupIssue:
    return SetupIssue(
        issue_name="syft_update_available",
        description=(
            "A new release of Syft is available: "
            + f"{str(current_version)} -> {str(new_version)}."
        ),
        command=f"pip install syft=={new_version}",
        solution="You can upgrade Syft with pip.",
    )


def hagrid_update_available(
    current_version: Version, new_version: Version
) -> SetupIssue:
    return SetupIssue(
        issue_name="hagrid_update_available",
        description=(
            "A new release of HAGrid is available: "
            + f"{str(current_version)} -> {str(new_version)}."
        ),
        command=f"pip install -U hagrid=={new_version}",
        solution="You can upgrade HAGrid with pip.",
    )


def python_version_unsupported() -> SetupIssue:
    return SetupIssue(
        issue_name="python_version_unsupported",
        description=(
            f"Syft supports Python >= {SYFT_MINIMUM_PYTHON_VERSION_STRING} "
            + f"and <= {SYFT_MAXIMUM_PYTHON_VERSION_STRING}"
        ),
        command="",
        solution="You must install a compatible version of Python",
    )


WINDOWS_JAXLIB_REPO = "https://whls.blob.core.windows.net/unstable/index.html"


def windows_jaxlib() -> SetupIssue:
    return SetupIssue(
        issue_name="windows_jaxlib",
        description="Windows Python Wheels for Jax are not available on PyPI yet",
        command=f"pip install jaxlib==0.3.14 -f {WINDOWS_JAXLIB_REPO}",
        solution="Windows users must install jaxlib before syft",
    )
