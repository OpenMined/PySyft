import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_APPS = [
    "https://github.com/OpenMined/logged_in",
    "https://github.com/OpenMined/inbox",
    "https://github.com/OpenMined/cpu_tracker_member",
    "https://github.com/OpenMined/DatasetLoader",
]


def download_github_repo(url: str, target_dir: str = None) -> Path:
    """Downloads and extracts a GitHub repository without git."""
    if not url.startswith(("http://", "https://")):
        raise ValueError("Invalid GitHub URL")

    repo_name = url.rstrip("/").split("/")[-1]
    target_dir = Path(target_dir or os.getcwd()) / repo_name

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        try:
            zip_path = tmp_path / "repo.zip"
            urllib.request.urlretrieve(f"{url}/archive/main.zip", zip_path)

            with zipfile.ZipFile(zip_path) as zip_ref:
                extracted = tmp_path / zip_ref.namelist()[0].split("/")[0]
                zip_ref.extractall(tmp_path)
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                shutil.move(str(extracted), str(target_dir))

            return target_dir
        except Exception as e:
            print(f"Failed to download or extract {url}: {e}")


def clone_apps():
    apps = DEFAULT_APPS

    # this is needed for E2E or integration testing to only install only select apps
    # DO NOT MERGE IT WITH DEFAULT_APPS
    env_apps = os.getenv("SYFTBOX_DEFAULT_APPS", None)
    if env_apps:
        print(f"SYFTBOX_DEFAULT_APPS={env_apps}")
        apps = env_apps.strip().split(",")

    print("Installing", apps)

    # Iterate over the list and clone each repository
    for url in apps:
        download_github_repo(url)

    print("Done")


if __name__ == "__main__":
    current_directory = Path(os.getcwd())

    apps_directory = current_directory.parent
    os.chdir(apps_directory)
    clone_apps()
    shutil.rmtree(current_directory)
