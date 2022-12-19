# stdlib
from enum import Enum
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess  # nosec
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# third party
import git
import requests
import rich
from rich import console
from rich import progress
from rich.table import Table

# relative
from .cache import DEFAULT_BRANCH
from .mode import EDITABLE_MODE
from .mode import hagrid_root


class GitRemoteProgress(git.RemoteProgress):
    # CREDITS: https://splunktool.com/python-progress-bar-for-git-clone
    OP_CODES = [
        "BEGIN",
        "CHECKING_OUT",
        "COMPRESSING",
        "COUNTING",
        "END",
        "FINDING_SOURCES",
        "RECEIVING",
        "RESOLVING",
        "WRITING",
    ]
    OP_CODE_MAP = {
        getattr(git.RemoteProgress, _op_code): _op_code for _op_code in OP_CODES
    }

    def __init__(self) -> None:
        super().__init__()
        self.progressbar = progress.Progress(
            progress.SpinnerColumn(),
            # *progress.Progress.get_default_columns(),
            progress.TextColumn("[progress.description]{task.description}"),
            progress.BarColumn(),
            progress.TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            "eta",
            progress.TimeRemainingColumn(),
            progress.TextColumn("{task.fields[message]}"),
            console=console.Console(),
            transient=False,
        )
        self.progressbar.start()
        self.active_task = None

    def __del__(self) -> None:
        # logger.info("Destroying bar...")
        self.progressbar.stop()

    @classmethod
    def get_curr_op(cls, op_code: int) -> str:
        """Get OP name from OP code."""
        # Remove BEGIN- and END-flag and get op name
        op_code_masked = op_code & cls.OP_MASK
        return cls.OP_CODE_MAP.get(op_code_masked, "?").title()

    def update(
        self,
        op_code: int,
        cur_count: Union[str, float],
        max_count: Optional[Union[str, float]] = None,
        message: Optional[str] = None,
    ) -> None:
        # Start new bar on each BEGIN-flag
        if op_code & self.BEGIN:
            self.curr_op = self.get_curr_op(op_code)
            # logger.info("Next: %s", self.curr_op)
            self.active_task = self.progressbar.add_task(
                description=self.curr_op,
                total=max_count,
                message=message,
            )

        self.progressbar.update(
            task_id=self.active_task,
            completed=cur_count,
            message=message,
        )

        # End progress monitoring on each END-flag
        if op_code & self.END:
            # logger.info("Done: %s", self.curr_op)
            self.progressbar.update(
                task_id=self.active_task,
                message=f"[bright_black]{message}",
            )


class ProcessStatus(Enum):
    RUNNING = "[blue]Running"
    DONE = "[green]Done"
    FAILED = "[red]Failed"


def docker_desktop_memory() -> int:

    path = str(Path.home()) + "/Library/Group Containers/group.com.docker/settings.json"

    try:
        f = open(path, "r")
        out = f.read()
        f.close()
        return json.loads(out)["memoryMiB"]

    except Exception:  # nosec
        # docker desktop not found - probably running linux
        return -1


def asset_path() -> os.PathLike:
    return Path(hagrid_root()) / "hagrid"


def manifest_template_path() -> os.PathLike:
    return Path(asset_path()) / "manifest_template.yml"


def hagrid_cache_dir() -> os.PathLike:
    return Path("~/.hagrid").expanduser()


def repo_src_path() -> Path:
    if EDITABLE_MODE:
        return Path(os.path.abspath(Path(hagrid_root()) / "../../"))
    else:
        return Path(os.path.join(Path(hagrid_cache_dir()) / "PySyft"))


def grid_src_path() -> str:
    return str(repo_src_path() / "packages" / "grid")


def check_is_git(path: Path) -> bool:
    is_repo = False
    try:
        git.Repo(path)
        is_repo = True
    except Exception:  # nosec
        pass
    return is_repo


def is_gitpod() -> bool:
    return bool(os.environ.get("GITPOD_WORKSPACE_URL", None))


def gitpod_url(port: Optional[int] = None) -> str:
    workspace_url = os.environ.get("GITPOD_WORKSPACE_URL", "")
    if port:
        workspace_url = workspace_url.replace("https://", f"https://{port}-")
    return workspace_url


def get_git_repo() -> git.Repo:
    # relative
    from .art import RichEmoji

    OK_EMOJI = RichEmoji("white_heavy_check_mark").to_str()

    is_git = check_is_git(path=repo_src_path())
    console = rich.get_console()
    if not EDITABLE_MODE and not is_git:
        github_repo = "OpenMined/PySyft.git"
        git_url = f"https://github.com/{github_repo}"

        print(f"Fetching Syft + Grid Source from {git_url} to {repo_src_path()}")
        try:
            repo_branch = DEFAULT_BRANCH
            repo_path = repo_src_path()

            if repo_path.exists():
                shutil.rmtree(str(repo_path))

            git.Repo.clone_from(
                git_url,
                str(repo_path),
                single_branch=False,
                b=repo_branch,
                progress=GitRemoteProgress(),
            )
            console.print(f"{OK_EMOJI} Fetched PySyft repo.")
        except Exception as e:  # nosec
            print(f"Failed to clone {git_url} to {repo_src_path()} with error: {e}")
    return git.Repo(repo_src_path())


def update_repo(repo: git.Repo, branch: str) -> None:
    # relative
    from .art import RichEmoji

    OK_EMOJI = RichEmoji("white_heavy_check_mark").to_str()
    console = rich.get_console()
    if not EDITABLE_MODE:
        with console.status("Updating hagrid") as console_status:

            console_status.update(f"[bold blue]Updating HAGrid from branch: {branch}")
            try:
                if repo.is_dirty():
                    repo.git.reset("--hard")
                repo.git.checkout(branch)
                repo.remotes.origin.pull()
                console.print(f"{OK_EMOJI} Updated HAGrid from branch: {branch}")
            except Exception as e:
                print(f"Error checking out branch {branch}.", e)


def commit_hash() -> str:
    try:
        repo = get_git_repo()
        sha = repo.head.commit.hexsha
        return sha
    except Exception as e:
        print("failed to get repo sha", e)
        return "unknown"


def use_branch(branch: str) -> None:
    if not EDITABLE_MODE:
        print(f"Using HAGrid from branch: {branch}")
        repo = get_git_repo()
        try:
            if repo.is_dirty():
                repo.git.reset("--hard")
            repo.git.checkout(branch)
            repo.remotes.origin.pull()
        except Exception as e:
            print(f"Error checking out branch {branch}.", e)


def should_provision_remote(
    username: Optional[str], password: Optional[str], key_path: Optional[str]
) -> bool:
    is_remote = username is not None or password is not None or key_path is not None
    if username and password or username and key_path:
        return is_remote
    if is_remote:
        raise Exception("--username requires either --password or --key-path")
    return is_remote


def name_tag(name: str) -> str:
    return hashlib.sha256(name.encode("utf8")).hexdigest()


def find_available_port(host: str, port: int, search: bool = False) -> int:
    port_available = False
    while not port_available:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result_of_check = sock.connect_ex((host, port))

            if result_of_check != 0:
                port_available = True
                break
            else:
                if search:
                    port += 1
                else:
                    break

        except Exception as e:
            print(f"Failed to check port {port}. {e}")
    sock.close()

    if search is False and port_available is False:
        error = (
            f"{port} is in use, either free the port or "
            + f"try: {port}+ to auto search for a port"
        )
        raise Exception(error)
    return port


def get_version_module() -> Tuple[str, str]:
    try:
        version_file_path = f"{grid_src_path()}/VERSION"
        loader = importlib.machinery.SourceFileLoader("VERSION", version_file_path)
        spec = importlib.util.spec_from_loader(loader.name, loader)
        if spec:
            version_module = importlib.util.module_from_spec(spec)
            loader.exec_module(version_module)
            version = version_module.get_version()
            hash = version_module.get_hash()
            return (version, hash)
    except Exception as e:
        print(f"Failed to retrieve versions from: {version_file_path}. {e}")
    return ("unknown", "unknown")


# Check base route of an IP address
def check_host(ip: str, silent: bool = False) -> bool:
    try:
        socket.gethostbyname(ip)
        return True
    except Exception as e:
        if not silent:
            print(f"Failed to resolve host {ip}. {e}")
        return False


# Check status of login page
def check_login_page(ip: str, timeout: int = 30, silent: bool = False) -> bool:
    try:
        url = f"http://{ip}/login"
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            return True
        else:
            return False
    except Exception as e:
        if not silent:
            print(f"Failed to check login page {ip}. {e}")
        return False


# Check api metadata
def check_api_metadata(ip: str, timeout: int = 30, silent: bool = False) -> bool:
    try:
        url = f"http://{ip}/api/v1/syft/metadata"
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            return True
        else:
            return False
    except Exception as e:
        if not silent:
            print(f"Failed to check api metadata {ip}. {e}")
        return False


def save_vm_details_as_json(username: str, password: str, process_list: List) -> None:
    """Saves the launched hosts details as json."""

    host_ip_details: List = []

    # file path to save host details
    dir_path = os.path.expanduser("~/.hagrid")
    os.makedirs(dir_path, exist_ok=True)
    file_path = f"{dir_path}/host_ips.json"

    for ip_address, _, jupyter_token in process_list:
        _data = {
            "username": username,
            "password": password,
            "ip_address": ip_address,
            "jupyter_token": jupyter_token,
        }
        host_ip_details.append(_data)

    # save host details
    with open(file_path, "w") as fp:
        json.dump({"host_ips": host_ip_details}, fp)

    print(f"Saved vm details at: {file_path}")


def generate_user_table(username: str, password: str) -> Union[Table, str]:
    if not username and not password:
        return ""

    table = Table(title="Virtual Machine Credentials")
    table.add_column("Username")
    table.add_column("Password")

    table.add_row(f"[green]{username}", f"[green]{password}")

    return table


def get_process_status(process: subprocess.Popen) -> str:
    poll_status = process.poll()
    if poll_status is None:
        return ProcessStatus.RUNNING.value
    elif poll_status != 0:
        return ProcessStatus.FAILED.value
    else:
        return ProcessStatus.DONE.value


def generate_process_status_table(process_list: List) -> Tuple[Table, bool]:
    """Generate a table to show the status of the processes being exected.

    Args:
        process_list (list): each item in the list
        is a tuple of ip_address, process and jupyter token

    Returns:
        Tuple[Table, bool]: table of process status and flag to indicate if all processes are executed.
    """

    process_statuses: List[str] = []
    lines_to_display = 5  # Number of lines to display as output

    table = Table(title="Virtual Machine Status")
    table.add_column("PID", style="cyan")
    table.add_column("IpAddress", style="magenta")
    table.add_column("Status")
    table.add_column("Jupyter Token", style="white on black")
    table.add_column("Log", overflow="fold", no_wrap=False)

    for ip_address, process, jupyter_token in process_list:
        process_status = get_process_status(process)

        process_statuses.append(process_status)

        process_log = []
        if process_status == ProcessStatus.FAILED.value:
            process_log += process.stderr.readlines(lines_to_display)
        else:
            process_log += process.stdout.readlines(lines_to_display)

        process_log_str = "\n".join(log.decode("utf-8") for log in process_log)
        process_log_str = process_log_str if process_log else "-"

        table.add_row(
            f"{process.pid}",
            f"{ip_address}",
            f"{process_status}",
            f"{jupyter_token}",
            f"{process_log_str}",
        )

    processes_completed = ProcessStatus.RUNNING.value not in process_statuses

    return table, processes_completed


def check_jupyter_server(
    host_ip: str, wait_time: int = 5, silent: bool = False
) -> bool:
    if not silent:
        print(f"Checking Jupyter Server at VM {host_ip} is up")

    try:
        url = f"http://{host_ip}:8888/"
        response = requests.get(url, timeout=wait_time)
        if response.status_code == 200:
            return True
        else:
            return False
    except Exception as e:
        if not silent:
            print(f"Failed to check jupyter server status {host_ip}. {e}")
        return False


GIT_REPO = get_git_repo()
GRID_SRC_VERSION = get_version_module()
GRID_SRC_PATH = grid_src_path()
