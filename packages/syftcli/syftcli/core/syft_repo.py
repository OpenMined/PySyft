# future
from __future__ import annotations

# stdlib
from functools import lru_cache
from typing import List

# third party
import requests
import yaml

REPO = "OpenMined/PySyft"
REPO_API_URL = f"https://api.github.com/repos/{REPO}"
REPO_DL_URL = f"https://github.com/{REPO}/releases/download"

ASSET_MANIFEST = "manifest_template.yml"


class SyftRepo:
    @staticmethod
    @lru_cache(maxsize=None)
    def releases() -> List[dict]:
        url = REPO_API_URL + "/releases"
        releases = requests.get(url).json()
        return [rel for rel in releases if rel.get("tag_name", "").startswith("v")]

    @staticmethod
    def latest_version() -> str:
        latest_release = SyftRepo.releases()[0]
        return latest_release["tag_name"]

    @staticmethod
    def all_versions() -> List[str]:
        return [rel["tag_name"] for rel in SyftRepo.releases() if rel.get("tag_name")]

    @staticmethod
    @lru_cache(maxsize=None)
    def get_manifest_template(rel_ver: str) -> dict:
        """
        Returns the manifest_template.yml for a given release version

        Args:
            rel_ver: str - OpenMined/Syft github release version. Must start with "v"
        """

        results = SyftRepo.get_asset(rel_ver, ASSET_MANIFEST)
        parsed = yaml.safe_load(results.text)
        return parsed

    @staticmethod
    def get_asset(rel_ver: str, asset_name: str) -> requests.Response:
        url = REPO_DL_URL + f"/{rel_ver}/{asset_name}"
        return requests.get(url)
