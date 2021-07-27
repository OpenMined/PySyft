# stdlib
import hashlib
import os
from pathlib import Path
import site
import subprocess

# third party
import git
import names
import requests

# relative
from .deps import MissingDependency

DOCKER_ERROR = """
Instructions for v2 beta can be found here:
You are running an old verion of docker, possibly on Linux. You need to install v2 beta.

https://www.rockyourcode.com/how-to-install-docker-compose-v2-on-linux-2021/

At the time of writing this, if you are on linux you need to run the following:

mkdir -p ~/.docker/cli-plugins
curl -sSL https://github.com/docker/compose-cli/releases/download/v2.0.0-beta.5/docker-compose-linux-amd64 \
-o ~/.docker/cli-plugins/docker-compose
chmod +x ~/.docker/cli-plugins/docker-compose

ALERT: you may need to run the following command to make sure you can run without sudo.

echo $USER              //(should return your username)
sudo usermod -aG docker $USER

... now LOG ALL THE WAY OUT!!!

...and then you should be good to go. You can check your installation by running:

docker compose version
"""


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
    return installed_as_editable


def repo_src_path() -> os.PathLike:
    if EDITABLE_MODE:
        return Path(os.path.abspath(Path(hagrid_root()) / "../../"))
    else:
        return Path(hagrid_root()) / "hagrid" / "PySyft"


def grid_src_path() -> str:
    return str(repo_src_path() / "packages" / "grid")


def check_is_git(path: os.PathLike) -> bool:
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
        github_repo = "OpenMined/PySyft"
        git_url = f"https://github.com/{github_repo}"
        print(f"Fetching Syft + Grid Source from {git_url} to {repo_src_path()}")
        try:
            repo_branch = "demo_strike_team_branch_4"
            git.Repo.clone_from(
                git_url, repo_src_path(), single_branch=True, b=repo_branch
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


EDITABLE_MODE = is_editable_mode()
GRID_SRC_PATH = grid_src_path()
GIT_REPO = get_git_repo()


repo_branch = "demo_strike_team_branch_4"
update_repo(repo=GIT_REPO, branch=repo_branch)


def should_provision_remote(username, password, key_path) -> bool:
    is_remote = username is not None or password is not None or key_path is not None
    if username and password or username and key_path:
        return is_remote
    if is_remote:
        raise Exception("--username requires either --password or --key_path")
    return is_remote


def name_tag(name: str) -> str:
    return hashlib.md5(name.encode("utf8")).hexdigest()


def pre_process_name(name: list, node_type: str) -> str:
    #  concatenate name's list of words into string
    _name = ""
    for word in name:
        _name += word + " "
    name = _name[:-1]

    if name == "" or name == ():
        name = "The " + names.get_full_name() + " " + node_type.capitalize()

    return name


def pre_process_keep_db(keep_db, tag) -> bool:
    if isinstance(keep_db, str):
        keep_db = True if keep_db.lower() == "true" else False
    return keep_db


def find_available_port(host: str, port: int, search: bool = False) -> bool:
    port_available = False
    while not port_available:
        try:
            requests.get("http://" + host + ":" + str(port))
            if search:
                print(
                    str(port)
                    + " doesn't seem to be available... trying "
                    + str(port + 1)
                )
                port = port + 1
            else:
                break
        except requests.ConnectionError:
            port_available = True
    if search is False and port_available is False:
        error = (
            f"{port} is in use, either free the port or "
            + f"try: {port}+ to auto search for a port"
        )
        raise Exception(error)
    return port


def check_docker_version() -> str:
    result = os.popen("docker compose version", "r").read()

    if "version" in result:
        version = result.split()[-1]
    else:
        print("This may be a linux machine, either that or docker compose isn't s")
        print("Result:" + result)
        out = subprocess.run(["docker", "compose"], capture_output=True, text=True)
        if "'compose' is not a docker command" in out.stderr:
            raise MissingDependency(DOCKER_ERROR)

    return version
