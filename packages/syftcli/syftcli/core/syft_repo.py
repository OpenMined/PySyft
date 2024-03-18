# stdlib
from functools import cache
from pathlib import Path
import shutil
from typing import Any

# third party
import requests
import yaml

REPO = "OpenMined/PySyft"
REPO_API_URL = f"https://api.github.com/repos/{REPO}"
REPO_DL_URL = f"https://github.com/{REPO}/releases/download"


class SyftRepo:
    class Assets:
        MANIFEST = "manifest.yml"
        PODMAN_CONFIG = "podman_config.tar.gz"
        DOCKER_CONFIG = "docker_config.tar.gz"

    @staticmethod
    @cache
    def releases() -> list[dict]:
        url = REPO_API_URL + "/releases"
        response = requests.get(url)
        response.raise_for_status()
        releases = response.json()
        return [rel for rel in releases if rel.get("tag_name", "").startswith("v")]

    @staticmethod
    @cache
    def prod_releases() -> list[dict]:
        return [rel for rel in SyftRepo.releases() if not rel.get("prerelease")]

    @staticmethod
    @cache
    def beta_releases() -> list[dict]:
        return [rel for rel in SyftRepo.releases() if rel.get("prerelease")]

    @staticmethod
    def latest_version(beta: bool = False) -> str:
        if beta:
            latest_release = SyftRepo.beta_releases()[0]
        else:
            latest_release = SyftRepo.prod_releases()[0]
        return latest_release["tag_name"]

    @staticmethod
    def all_versions() -> list[str]:
        return [rel["tag_name"] for rel in SyftRepo.releases() if rel.get("tag_name")]

    @staticmethod
    @cache
    def get_manifest(rel_ver: str) -> dict:
        """
        Returns the manifest_template.yml for a given release version

        Args:
            rel_ver: str - OpenMined/Syft github release version. Must start with "v"
        """

        results = SyftRepo.get_asset(rel_ver, SyftRepo.Assets.MANIFEST)
        parsed = yaml.safe_load(results.text)
        return parsed

    @staticmethod
    def download_asset(asset_name: str, rel_ver: str, dl_dir: str) -> Path:
        asset_path = Path(dl_dir, asset_name)
        resp = SyftRepo.get_asset(rel_ver, asset_name, stream=True)

        with open(asset_path, "wb") as fp:
            shutil.copyfileobj(resp.raw, fp)

        return asset_path

    @staticmethod
    def get_asset(rel_ver: str, asset_name: str, **kwargs: Any) -> requests.Response:
        url = REPO_DL_URL + f"/{rel_ver}/{asset_name}"
        response = requests.get(url, **kwargs)
        response.raise_for_status()
        return response
