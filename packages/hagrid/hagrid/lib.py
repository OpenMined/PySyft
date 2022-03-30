# stdlib
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import site
import socket
import subprocess
from typing import Optional
from typing import Tuple

# third party
import git

# relative
from .cache import DEFAULT_BRANCH
from .deps import MissingDependency
from .deps import is_windows

DOCKER_ERROR = """
You are running an old version of docker, possibly on Linux. You need to install v2.
At the time of writing this, if you are on linux you need to run the following:

DOCKER_COMPOSE_VERSION=v2.3.4
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


def docker_desktop_memory() -> int:

    path = str(Path.home()) + "/Library/Group Containers/group.com.docker/settings.json"

    try:
        f = open(path, "r")
        out = f.read()
        f.close()
        return json.loads(out)["memoryMiB"]

    except Exception:
        # docker desktop not found - probably running linux
        return -1


def hagrid_root() -> str:
    return os.path.abspath(str(Path(__file__).parent.parent))


def asset_path() -> os.PathLike:
    return Path(hagrid_root()) / "hagrid"


def is_editable_mode() -> bool:
    current_package_root = hagrid_root()

    installed_as_editable = False
    sitepackages_dirs = site.getsitepackages()
    # check all site-packages returned if they have a hagrid.egg-link
    for sitepackages_dir in sitepackages_dirs:
        egg_link_file = Path(sitepackages_dir) / "hagrid.egg-link"
        try:
            linked_folder = egg_link_file.read_text()
            # if the current code is in the same path as the egg-link its -e mode
            installed_as_editable = current_package_root in linked_folder
            break
        except Exception:
            pass

    if os.path.exists(Path(current_package_root) / "hagrid.egg-info"):
        installed_as_editable = True

    return installed_as_editable


def repo_src_path() -> Path:
    if EDITABLE_MODE:
        return Path(os.path.abspath(Path(hagrid_root()) / "../../"))
    else:
        return Path(hagrid_root()) / "hagrid" / "PySyft"


def grid_src_path() -> str:
    return str(repo_src_path() / "packages" / "grid")


def check_is_git(path: Path) -> bool:
    is_repo = False
    try:
        git.Repo(path)
        is_repo = True
    except Exception:
        print(f"{path} is not a git repo!")
    return is_repo


def get_git_repo() -> git.Repo:
    is_git = check_is_git(path=repo_src_path())
    if not EDITABLE_MODE and not is_git:
        github_repo = "OpenMined/PySyft.git"
        git_url = f"https://github.com/{github_repo}"
        print(f"Fetching Syft + Grid Source from {git_url} to {repo_src_path()}")
        try:
            repo_branch = DEFAULT_BRANCH
            git.Repo.clone_from(
                git_url, repo_src_path(), single_branch=False, b=repo_branch
            )
        except Exception:
            print(f"Failed to clone {git_url} to {repo_src_path()}")
    return git.Repo(repo_src_path())


def update_repo(repo: git.Repo, branch: str) -> None:
    if not EDITABLE_MODE:
        print(f"Updating HAGrid from branch: {branch}")
        try:
            if repo.is_dirty():
                repo.git.reset("--hard")
            repo.git.checkout(branch)
            repo.remotes.origin.pull()
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


EDITABLE_MODE = is_editable_mode()
GRID_SRC_PATH = grid_src_path()
GIT_REPO = get_git_repo()


repo_branch = DEFAULT_BRANCH
update_repo(repo=GIT_REPO, branch=repo_branch)


def should_provision_remote(
    username: Optional[str], password: Optional[str], key_path: Optional[str]
) -> bool:
    is_remote = username is not None or password is not None or key_path is not None
    if username and password or username and key_path:
        return is_remote
    if is_remote:
        raise Exception("--username requires either --password or --key_path")
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


def check_docker_version() -> Optional[str]:
    if is_windows():
        return "N/A"  # todo fix to work with windows
    result = os.popen("docker compose version", "r").read()
    version = None
    if "version" in result:
        version = result.split()[-1]
    else:
        print("This may be a linux machine, either that or docker compose isn't s")
        print("Result:" + result)
        out = subprocess.run(["docker", "compose"], capture_output=True, text=True)
        if "'compose' is not a docker command" in out.stderr:
            raise MissingDependency(DOCKER_ERROR)

    return version


def get_version_module() -> Tuple[str, str]:
    try:
        version_file_path = f"{GRID_SRC_PATH}/VERSION"
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


GRID_SRC_VERSION = get_version_module()
